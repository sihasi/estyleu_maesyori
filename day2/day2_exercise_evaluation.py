"""Shared execution and offline retrieval evaluation for the Day 2 exercise.

The notebooks intentionally keep only the learner implementation, the calls to
this module, and result display.  Keeping the execution and scoring contract
here makes the generated exercise variants use exactly the same evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Protocol

import pandas as pd


TOP_K = 5
MAX_CONTEXT_CHARS = 2_500
SOURCE_SUFFIXES = frozenset({".docx", ".xlsx", ".pptx", ".pdf"})
EXPECTED_GROUP_COUNTS = {
    "基本確認": 4,
    "単一文書": 12,
    "複数文書": 12,
    "文脈復元": 8,
    "語彙差": 4,
}


class SearchEngine(Protocol):
    """Minimal contract expected from a learner's search implementation."""

    def search(self, query: str) -> list[str]: ...


@dataclass(frozen=True)
class EvaluationResult:
    """Question-level and aggregate dataframes returned by one evaluation run."""

    questions: pd.DataFrame
    results: pd.DataFrame
    summary: pd.DataFrame
    by_group: pd.DataFrame


def source_paths(data_dir: Path) -> list[Path]:
    """Return the four supported source formats in stable name order."""
    return [
        source_path
        for source_path in sorted(data_dir.iterdir())
        if source_path.suffix.lower() in SOURCE_SUFFIXES
    ]


def run_extraction(
    data_dir: Path, extract_document: Callable[[Path], list[Any]]
) -> list[Any]:
    """Apply the learner's extractor and validate each input's result."""
    extracted_items: list[Any] = []
    for source_path in source_paths(data_dir):
        items = extract_document(source_path)
        if not isinstance(items, list) or not items:
            raise TypeError(
                f"extract_document({source_path.name})は空でないlist[Any]を返してください"
            )
        extracted_items.extend(items)
        print(f"{source_path.name}: {len(items)}件を抽出")
    return extracted_items


def validate_search_engine(search_engine: SearchEngine) -> None:
    """Fail early when the returned value does not implement the search contract."""
    if not callable(getattr(search_engine, "search", None)):
        raise TypeError(
            "build_search_engine は最大5件を返す search(query) を持つ検索器を返してください。"
        )


def load_questions(data_dir: Path) -> pd.DataFrame:
    """Load and validate the fixed forty-question evaluation set."""
    questions = pd.read_csv(data_dir / "evaluation_questions.csv")
    if len(questions) != 40:
        raise AssertionError(f"評価データは40問を想定しています: {len(questions)}問")
    if questions["question_id"].duplicated().any():
        raise AssertionError("question_idが重複しています")
    if questions["evaluation_group"].value_counts().to_dict() != EXPECTED_GROUP_COUNTS:
        raise AssertionError("評価群の件数が想定と異なります")
    if "required_facts" not in questions.columns or questions["required_facts"].isna().any():
        raise AssertionError("required_factsが不足しています")
    return questions


def split_required_facts(value: object) -> list[str]:
    return [fact.strip() for fact in str(value or "").split("|") if fact.strip()]


def limit_search_results(ranking: list[str]) -> list[str]:
    """Keep rank order while limiting five results to 2,500 total characters."""
    remaining_chars = MAX_CONTEXT_CHARS
    limited_results: list[str] = []
    for result in ranking:
        if not isinstance(result, str):
            raise TypeError("検索器のsearchはlist[str]を返してください")
        if remaining_chars <= 0:
            break
        text = result[:remaining_chars]
        if text:
            limited_results.append(text)
            remaining_chars -= len(text)
    return limited_results


def required_fact_gains(
    ranking: list[str], required_facts: object, k: int = TOP_K
) -> tuple[list[int], int]:
    """Return the number of newly-covered required facts at every rank."""
    facts = split_required_facts(required_facts)
    normalized_facts = [re.sub(r"\s+", "", fact).casefold() for fact in facts]
    matched_fact_indexes: set[int] = set()
    gains: list[int] = []
    for result in ranking[:k]:
        if not isinstance(result, str):
            raise TypeError("検索器のsearchはlist[str]を返してください")
        text = re.sub(r"\s+", "", result).casefold()
        if not text:
            raise ValueError("検索結果の文字列は空にできません")
        newly_matched = {
            index
            for index, fact in enumerate(normalized_facts)
            if index not in matched_fact_indexes and fact in text
        }
        matched_fact_indexes.update(newly_matched)
        gains.append(len(newly_matched))
    return gains, len(facts)


def required_fact_recall_at_k(
    ranking: list[str], required_facts: object, k: int = TOP_K
) -> float:
    gains, fact_count = required_fact_gains(ranking, required_facts, k)
    return sum(gains) / fact_count if fact_count else 0.0


def required_fact_ndcg_at_k(
    ranking: list[str], required_facts: object, k: int = TOP_K
) -> float:
    gains, fact_count = required_fact_gains(ranking, required_facts, k)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    return dcg / fact_count if fact_count else 0.0


def evaluate_retrieval(
    data_dir: Path, search_engine: SearchEngine, pipeline_name: str
) -> EvaluationResult:
    """Evaluate one search pipeline with the shared Recall@5/nDCG@5 contract."""
    validate_search_engine(search_engine)
    questions = load_questions(data_dir)
    evaluation_rows: list[dict[str, object]] = []
    for _, question_row in questions.iterrows():
        ranking = search_engine.search(question_row["question"])
        if not isinstance(ranking, list):
            raise TypeError("検索器のsearchはlist[str]を返してください")
        if len(ranking) > TOP_K:
            raise AssertionError(f"検索器のsearchは最大{TOP_K}件を返してください")
        ranking = limit_search_results(ranking)
        if not split_required_facts(question_row["required_facts"]):
            raise ValueError("required_factsが空です")

        evaluation_rows.append(
            {
                "question_id": question_row["question_id"],
                "evaluation_group": question_row["evaluation_group"],
                "question": question_row["question"],
                "pipeline_name": pipeline_name,
                "result_count": len(ranking),
                "result_text_chars": sum(len(result) for result in ranking),
                "max_result_text_chars": max(
                    (len(result) for result in ranking), default=0
                ),
                "recall_at_5": required_fact_recall_at_k(
                    ranking, question_row["required_facts"]
                ),
                "ndcg_at_5": required_fact_ndcg_at_k(
                    ranking, question_row["required_facts"]
                ),
            }
        )

    results = pd.DataFrame(evaluation_rows)
    if len(results) != len(questions):
        raise AssertionError("各質問を1回ずつ評価できていません")
    summary = pd.DataFrame(
        [
            {
                "pipeline_name": pipeline_name,
                "questions": len(results),
                "recall_at_5": results["recall_at_5"].mean(),
                "ndcg_at_5": results["ndcg_at_5"].mean(),
                "mean_result_count": results["result_count"].mean(),
                "mean_result_text_chars": results["result_text_chars"].mean(),
                "mean_max_result_text_chars": results[
                    "max_result_text_chars"
                ].mean(),
            }
        ]
    )
    by_group = (
        results.groupby("evaluation_group", as_index=False)
        .agg(
            questions=("question_id", "size"),
            recall_at_5=("recall_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            mean_result_count=("result_count", "mean"),
            mean_result_text_chars=("result_text_chars", "mean"),
            mean_max_result_text_chars=("max_result_text_chars", "mean"),
        )
    )
    by_group.insert(0, "pipeline_name", pipeline_name)
    return EvaluationResult(questions, results, summary, by_group)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import build_index, retrieve, make_grounded_answer


def sample_documents():
    return [
        {"name": "attendance_policy.txt", "text": "Students are expected to maintain at least 75 percent attendance in each course."},
        {"name": "library_rules.txt", "text": "Students may borrow up to three books. The loan period is fourteen days. Students must clear library dues before library clearance."},
        {"name": "exam_policy.txt", "text": "Students need a valid identity card and examination hall ticket. They must meet the attendance requirement for exam eligibility."},
        {"name": "calendar.txt", "text": "Final theory examinations begin on December 1."},
        {"name": "scholarship.txt", "text": "Scholarship applicants should check current eligibility criteria and submission deadlines with the scholarship office."},
    ]


def test_single_document_answer_has_source():
    vectorizer, chunks, matrix = build_index(sample_documents())
    results = retrieve("What is the minimum attendance requirement?", vectorizer, chunks, matrix)
    assert results
    answer = make_grounded_answer(results)
    assert "75 percent" in answer
    assert any(r["source"] == "attendance_policy.txt" for r in results)


def test_question_needing_two_documents_retrieves_both():
    vectorizer, chunks, matrix = build_index(sample_documents())
    results = retrieve("What do I need for exam eligibility and library clearance?", vectorizer, chunks, matrix, top_k=5)
    sources = {r["source"] for r in results}
    assert "exam_policy.txt" in sources
    assert "library_rules.txt" in sources


def test_unrelated_question_abstains():
    vectorizer, chunks, matrix = build_index(sample_documents())
    results = retrieve("What is the principal's personal phone number?", vectorizer, chunks, matrix)
    answer = make_grounded_answer(results)
    assert "couldn't find" in answer.lower() or "not find" in answer.lower()

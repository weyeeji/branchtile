"""Independent invariants: coverage, atomic tables, stable RRF and budgets."""
from retrieval_baseline import make_chunks, hybrid_rank, select_context


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_chunk_coverage_and_atomic_rows():
    text = "Intro\n|Year|USD|\n|---|---|\n|2024|100|\n|2025|200|\nEnd\n"
    chunks = make_chunks(text, CharTokenizer(), 30)
    assert "".join(text[c.source_start:c.source_end] for c in chunks) == text
    for row in ["|2024|100|", "|2025|200|"]:
        assert any(row in c.text for c in chunks)
    assert any(c.table_header_repeated for c in chunks)


def test_budget_never_truncates_selected_content():
    t = CharTokenizer()
    chunks = make_chunks("first\nsecond\nthird\n", t, 7)
    content, meta = select_context(chunks, [2, 0, 1], t, 14)
    assert meta["retrieved_tokens"] <= 14
    assert all(chunks[i].text in content for i in meta["selected_chunks"])
    assert meta["selected_chunks"] == sorted(meta["selected_chunks"])


def test_rrf_ties_use_original_position():
    chunks = make_chunks("same\nsame\nsame\n", CharTokenizer(), 5)
    order, _ = hybrid_rank(chunks, "absent", [0, 0, 0])
    assert order == [0, 1, 2]

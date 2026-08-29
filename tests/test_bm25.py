from semantra.bm25 import BM25


def test_bm25_prefers_matching_document():
    scores = BM25(["speaker has no sound", "battery will not charge"]).scores("no sound")
    assert scores[0] > scores[1]

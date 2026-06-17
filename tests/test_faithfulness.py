from syntony.metrics import faithfulness_score, label_counts, summarize
from syntony.schema import ClaimVerdict, Evidence, Label


def _verdict(label):
    evidence = []
    if label is not Label.UNSUPPORTED:
        evidence = [Evidence(source_span="x", score=0.9, verifier="t")]
    return ClaimVerdict(
        claim="c", label=label, evidence=evidence, confidence=0.9, verifier="t"
    )


def test_counts_always_have_three_keys():
    counts = label_counts([])
    assert counts == {"supported": 0, "contradicted": 0, "unsupported": 0}


def test_score_is_supported_fraction():
    claims = [
        _verdict(Label.SUPPORTED),
        _verdict(Label.SUPPORTED),
        _verdict(Label.CONTRADICTED),
        _verdict(Label.UNSUPPORTED),
    ]
    assert faithfulness_score(claims) == 0.5


def test_contradicted_and_unsupported_not_merged():
    claims = [_verdict(Label.CONTRADICTED), _verdict(Label.UNSUPPORTED)]
    counts = label_counts(claims)
    assert counts["contradicted"] == 1
    assert counts["unsupported"] == 1
    # Both lower the score, but they remain distinguishable.
    assert faithfulness_score(claims) == 0.0


def test_empty_is_vacuously_faithful():
    assert faithfulness_score([]) == 1.0


def test_summarize_pairs_score_and_counts():
    claims = [_verdict(Label.SUPPORTED), _verdict(Label.UNSUPPORTED)]
    score, counts = summarize(claims)
    assert score == 0.5
    assert counts["supported"] == 1

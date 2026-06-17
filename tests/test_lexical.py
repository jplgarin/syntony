import pytest

from syntony.schema import Label
from syntony.verify import LexicalVerifier

CTX = (
    "The trial enrolled 200 patients. "
    "The drug reduced mortality by 30 percent. "
    "The treatment was not effective for chronic pain."
)


def test_supported_claim_has_evidence():
    v = LexicalVerifier()
    verdict = v.verify("The trial enrolled 200 patients.", CTX)
    assert verdict.label is Label.SUPPORTED
    assert verdict.evidence
    assert "200 patients" in verdict.evidence[0].source_span
    assert verdict.evidence[0].verifier == "lexical"


def test_unsupported_when_context_silent():
    v = LexicalVerifier()
    verdict = v.verify("The company was founded in Berlin.", CTX)
    assert verdict.label is Label.UNSUPPORTED
    assert verdict.evidence == []


def test_negation_polarity_flips_to_contradicted():
    v = LexicalVerifier()
    verdict = v.verify("The treatment was effective for chronic pain.", CTX)
    assert verdict.label is Label.CONTRADICTED
    assert verdict.evidence


def test_matching_negation_is_supported():
    v = LexicalVerifier()
    verdict = v.verify("The treatment was not effective for chronic pain.", CTX)
    assert verdict.label is Label.SUPPORTED


def test_contraction_negation_handled():
    v = LexicalVerifier()
    ctx = "The system does not crash on startup."
    verdict = v.verify("The system doesn't crash on startup.", ctx)
    assert verdict.label is Label.SUPPORTED


def test_empty_claim_is_unsupported():
    v = LexicalVerifier()
    verdict = v.verify("the of to", CTX)  # all stopwords
    assert verdict.label is Label.UNSUPPORTED
    assert verdict.confidence == 0.5


def test_empty_context_is_unsupported():
    v = LexicalVerifier()
    verdict = v.verify("Something happened here.", "")
    assert verdict.label is Label.UNSUPPORTED


def test_threshold_validation():
    with pytest.raises(ValueError):
        LexicalVerifier(support_threshold=0.0)
    with pytest.raises(ValueError):
        LexicalVerifier(unigram_weight=1.5)


def test_confidence_is_bounded():
    v = LexicalVerifier()
    for claim in ["The trial enrolled 200 patients.", "Totally unrelated text."]:
        verdict = v.verify(claim, CTX)
        assert 0.0 <= verdict.confidence <= 1.0

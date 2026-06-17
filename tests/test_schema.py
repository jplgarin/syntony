import pytest
from pydantic import ValidationError

from syntony.schema import ClaimVerdict, EvalCase, EvalResult, Evidence, Label


def test_label_str_is_value():
    assert str(Label.SUPPORTED) == "supported"
    assert Label("contradicted") is Label.CONTRADICTED


def test_evidence_score_bounds():
    Evidence(source_span="x", score=0.0, verifier="lexical")
    Evidence(source_span="x", score=1.0, verifier="lexical")
    with pytest.raises(ValidationError):
        Evidence(source_span="x", score=1.5, verifier="lexical")


def test_evidence_is_frozen():
    ev = Evidence(source_span="x", score=0.5, verifier="lexical")
    with pytest.raises(ValidationError):
        ev.score = 0.9


def test_supported_requires_evidence():
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim="c", label=Label.SUPPORTED, evidence=[], confidence=0.9,
            verifier="lexical",
        )


def test_contradicted_requires_evidence():
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim="c", label=Label.CONTRADICTED, evidence=[], confidence=0.9,
            verifier="lexical",
        )


def test_unsupported_allows_empty_evidence():
    v = ClaimVerdict(
        claim="c", label=Label.UNSUPPORTED, evidence=[], confidence=0.4,
        verifier="lexical",
    )
    assert v.evidence == []


def test_eval_result_roundtrip():
    case = EvalCase(output_text="a", source_context="b", metadata={"k": 1})
    verdict = ClaimVerdict(
        claim="a",
        label=Label.SUPPORTED,
        evidence=[Evidence(source_span="b", score=0.9, verifier="lexical")],
        confidence=0.9,
        verifier="lexical",
    )
    result = EvalResult(
        case=case,
        claims=[verdict],
        faithfulness_score=1.0,
        counts={"supported": 1, "contradicted": 0, "unsupported": 0},
        verifier_name="lexical",
    )
    again = EvalResult.model_validate_json(result.model_dump_json())
    assert again.claims[0].label is Label.SUPPORTED
    assert again.case.metadata == {"k": 1}

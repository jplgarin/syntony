"""NLI verifier tests.

These never download a model and never require transformers to be installed.
The pipeline is either injected directly (``verifier._pipeline``) or simulated
through ``sys.modules`` so the load-failure paths can be exercised offline.
"""

import sys

import pytest

from syntony.schema import Label
from syntony.verify.nli import NLIError, NLIVerifier, _normalize_nli_label


class FakePipe:
    """Stand-in for a transformers text-classification pipeline.

    Maps a premise (context sentence) to a fixed distribution over the three
    NLI classes, so a test can control exactly which sentence entails or
    contradicts the hypothesis.
    """

    def __init__(self, by_premise):
        self.by_premise = by_premise
        self.calls = []

    def __call__(self, payload):
        premise = payload["text"]
        self.calls.append(payload)
        scores = self.by_premise.get(premise, {"neutral": 1.0})
        return [
            {"label": "ENTAILMENT", "score": scores.get("entailment", 0.0)},
            {"label": "NEUTRAL", "score": scores.get("neutral", 0.0)},
            {"label": "CONTRADICTION", "score": scores.get("contradiction", 0.0)},
        ]


CTX = "The drug reduced mortality. The trial was double blind."


def test_normalize_labels():
    assert _normalize_nli_label("ENTAILMENT") == "entailment"
    assert _normalize_nli_label("LABEL_0 contradiction") == "contradiction"
    assert _normalize_nli_label("neutral") == "neutral"
    assert _normalize_nli_label("garbage") is None


def test_supported_picks_entailing_sentence():
    v = NLIVerifier()
    v._pipeline = FakePipe(
        {
            "The drug reduced mortality.": {"entailment": 0.95, "neutral": 0.05},
            "The trial was double blind.": {"neutral": 0.9, "entailment": 0.1},
        }
    )
    verdict = v.verify("The drug lowered deaths.", CTX)
    assert verdict.label is Label.SUPPORTED
    assert verdict.evidence[0].source_span == "The drug reduced mortality."
    assert verdict.confidence == pytest.approx(0.95)


def test_contradicted_maps_from_contradiction():
    v = NLIVerifier()
    v._pipeline = FakePipe(
        {"The drug reduced mortality.": {"contradiction": 0.88, "neutral": 0.12}}
    )
    verdict = v.verify("The drug increased mortality.", CTX)
    assert verdict.label is Label.CONTRADICTED
    assert verdict.evidence[0].source_span == "The drug reduced mortality."


def test_neutral_maps_to_unsupported_no_evidence():
    v = NLIVerifier()
    v._pipeline = FakePipe({})  # everything defaults to neutral
    verdict = v.verify("Unrelated claim.", CTX)
    assert verdict.label is Label.UNSUPPORTED
    assert verdict.evidence == []


def test_below_threshold_is_unsupported():
    v = NLIVerifier(decision_threshold=0.7)
    v._pipeline = FakePipe(
        {"The drug reduced mortality.": {"entailment": 0.4, "neutral": 0.6}}
    )
    verdict = v.verify("The drug lowered deaths.", CTX)
    assert verdict.label is Label.UNSUPPORTED


def test_empty_context_short_circuits():
    v = NLIVerifier()
    v._pipeline = FakePipe({})
    verdict = v.verify("Anything.", "")
    assert verdict.label is Label.UNSUPPORTED
    assert v._pipeline.calls == []  # pipeline never touched


def test_missing_transformers_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    v = NLIVerifier()
    with pytest.raises(NLIError) as exc:
        v.verify("claim", "Some context sentence.")
    assert "syntony[nli]" in str(exc.value)


def test_model_load_failure_raises_nlierror(monkeypatch):
    import types

    fake = types.ModuleType("transformers")

    def boom(*args, **kwargs):
        raise OSError("model not found")

    fake.pipeline = boom
    monkeypatch.setitem(sys.modules, "transformers", fake)
    v = NLIVerifier(model_name="does/not-exist")
    with pytest.raises(NLIError) as exc:
        v.verify("claim", "Some context sentence.")
    assert "does/not-exist" in str(exc.value)


def test_threshold_validation():
    with pytest.raises(ValueError):
        NLIVerifier(decision_threshold=0.0)

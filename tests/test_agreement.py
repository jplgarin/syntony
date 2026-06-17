import pytest

from syntony.agreement import agreement_report, cohens_kappa
from syntony.schema import ClaimVerdict, EvalCase, Evidence, Label
from syntony.verify import LexicalVerifier, Verifier


class ScriptedVerifier(Verifier):
    """Returns a preset label per claim, so agreement logic is testable."""

    def __init__(self, name, mapping, default=Label.UNSUPPORTED):
        self.name = name
        self.mapping = mapping
        self.default = default

    def verify(self, claim, context):
        label = self.mapping.get(claim, self.default)
        evidence = []
        if label is not Label.UNSUPPORTED:
            evidence = [Evidence(source_span=context[:10] or "x", score=0.8,
                                 verifier=self.name)]
        return ClaimVerdict(
            claim=claim, label=label, evidence=evidence, confidence=0.8,
            verifier=self.name,
        )


def test_cohens_kappa_perfect_and_chance():
    # Identical labels with two categories -> kappa 1.0.
    k, obs = cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert obs == 1.0
    assert k == pytest.approx(1.0)


def test_cohens_kappa_single_category_degenerate():
    k, obs = cohens_kappa(["a", "a"], ["a", "a"])
    assert obs == 1.0
    assert k == 1.0  # convention when chance agreement is total


def test_cohens_kappa_length_mismatch():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_report_flags_disagreement():
    case = EvalCase(
        output_text="Alpha is true. Beta is false.",
        source_context="Reference context text here.",
    )
    a = ScriptedVerifier(
        "a", {"Alpha is true.": Label.SUPPORTED, "Beta is false.": Label.SUPPORTED}
    )
    b = ScriptedVerifier(
        "b", {"Alpha is true.": Label.SUPPORTED, "Beta is false.": Label.CONTRADICTED}
    )
    report = agreement_report(case, [a, b])

    assert report.overall_agreement_rate == 0.5
    disagreed = report.disagreements
    assert len(disagreed) == 1
    assert disagreed[0].claim == "Beta is false."
    assert disagreed[0].labels == {"a": "supported", "b": "contradicted"}


def test_report_requires_two_verifiers():
    case = EvalCase(output_text="One claim.", source_context="ctx")
    with pytest.raises(ValueError):
        agreement_report(case, [LexicalVerifier()])


def test_report_kappa_present_for_pair():
    case = EvalCase(output_text="A. B. C.", source_context="ctx text")
    a = ScriptedVerifier("a", {}, default=Label.SUPPORTED)
    b = ScriptedVerifier("b", {}, default=Label.SUPPORTED)
    report = agreement_report(case, [a, b])
    assert len(report.pairwise_kappa) == 1
    assert report.pairwise_kappa[0].verifier_a == "a"


def test_report_empty_output():
    case = EvalCase(output_text="", source_context="ctx")
    a = ScriptedVerifier("a", {})
    b = ScriptedVerifier("b", {})
    report = agreement_report(case, [a, b])
    assert report.overall_agreement_rate == 1.0
    assert report.claim_agreements == []

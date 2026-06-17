from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from pydantic import BaseModel, Field

from ..decompose import Decomposer, SentenceDecomposer
from ..schema import ClaimVerdict, EvalCase
from ..verify import Verifier


class ClaimAgreement(BaseModel):
    """How the verifiers labelled one claim, and whether they concur."""

    claim: str
    labels: dict[str, str]
    verdicts: list[ClaimVerdict]
    agreed: bool


class PairKappa(BaseModel):
    """Cohen's kappa for one pair of verifiers."""

    verifier_a: str
    verifier_b: str
    kappa: float
    observed_agreement: float


class AgreementReport(BaseModel):
    """Cross-verifier agreement over a single case.

    The calibration view (``disagreements``) is the part that matters: those
    are the claims where the deterministic lexical verifier and the local NLI
    model see grounding differently, and they are exactly the claims worth a
    human's attention. Disagreement is signal.
    """

    verifiers: list[str]
    claim_agreements: list[ClaimAgreement]
    overall_agreement_rate: float
    pairwise_kappa: list[PairKappa] = Field(default_factory=list)

    @property
    def disagreements(self) -> list[ClaimAgreement]:
        return [c for c in self.claim_agreements if not c.agreed]


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> tuple[float, float]:
    """Cohen's kappa and observed agreement for two label sequences.

    Returns ``(kappa, observed_agreement)``. When chance agreement is 1.0
    (both raters used a single category for everything) kappa is undefined; we
    report 1.0 if they fully agreed and 0.0 otherwise, which is the
    conventional, least-surprising resolution.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        return 1.0, 1.0

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    categories = set(labels_a) | set(labels_b)
    expected = 0.0
    for category in categories:
        p_a = sum(1 for a in labels_a if a == category) / n
        p_b = sum(1 for b in labels_b if b == category) / n
        expected += p_a * p_b

    if expected >= 1.0:
        return (1.0 if observed >= 1.0 else 0.0), observed
    kappa = (observed - expected) / (1.0 - expected)
    return kappa, observed


def agreement_report(
    case: EvalCase,
    verifiers: Sequence[Verifier],
    decomposer: Decomposer | None = None,
) -> AgreementReport:
    """Run several verifiers over the same claims and compare them.

    Decomposition happens once so every verifier judges the identical set of
    claims; otherwise an "agreement" could just be an artefact of different
    claim boundaries.
    """
    if len(verifiers) < 2:
        raise ValueError("agreement requires at least two verifiers")

    decomposer = decomposer or SentenceDecomposer()
    claims = decomposer.decompose(case.output_text)

    # verifier name -> label per claim, used for the kappa computation.
    label_series: dict[str, list[str]] = {v.name: [] for v in verifiers}

    claim_agreements: list[ClaimAgreement] = []
    for claim in claims:
        verdicts = [v.verify(claim, case.source_context) for v in verifiers]
        labels = {v.name: verdict.label.value for v, verdict in zip(verifiers, verdicts)}
        for name, label in labels.items():
            label_series[name].append(label)
        agreed = len(set(labels.values())) == 1
        claim_agreements.append(
            ClaimAgreement(
                claim=claim,
                labels=labels,
                verdicts=verdicts,
                agreed=agreed,
            )
        )

    if claim_agreements:
        overall = sum(1 for c in claim_agreements if c.agreed) / len(claim_agreements)
    else:
        overall = 1.0

    pairwise: list[PairKappa] = []
    for a, b in combinations([v.name for v in verifiers], 2):
        kappa, observed = cohens_kappa(label_series[a], label_series[b])
        pairwise.append(
            PairKappa(
                verifier_a=a,
                verifier_b=b,
                kappa=round(kappa, 4),
                observed_agreement=round(observed, 4),
            )
        )

    return AgreementReport(
        verifiers=[v.name for v in verifiers],
        claim_agreements=claim_agreements,
        overall_agreement_rate=round(overall, 4),
        pairwise_kappa=pairwise,
    )

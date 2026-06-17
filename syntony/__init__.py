"""syntony: offline, traceable grounding and faithfulness verification.

The package import is deliberately cheap. Nothing here pulls in transformers or
torch; the NLI backend loads its model only when a verdict is requested.
"""

from __future__ import annotations

from .decompose import Decomposer, SentenceDecomposer
from .metrics import faithfulness_score, label_counts, summarize
from .schema import ClaimVerdict, EvalCase, EvalResult, Evidence, Label
from .verify import LexicalVerifier, NLIError, NLIVerifier, Verifier

__all__ = [
    "Decomposer",
    "SentenceDecomposer",
    "Verifier",
    "LexicalVerifier",
    "NLIVerifier",
    "NLIError",
    "Label",
    "Evidence",
    "ClaimVerdict",
    "EvalCase",
    "EvalResult",
    "faithfulness_score",
    "label_counts",
    "summarize",
    "evaluate",
]

__version__ = "0.1.0"


def evaluate(
    case: EvalCase,
    verifier: Verifier,
    decomposer: Decomposer | None = None,
) -> EvalResult:
    """Decompose, verify, and aggregate one case into a full result.

    This is the single entry point most callers want: it threads the same
    claims through one verifier and packages the verdicts, the scalar
    faithfulness score, and the per-label counts into a self-contained
    ``EvalResult``.
    """
    decomposer = decomposer or SentenceDecomposer()
    claims = decomposer.decompose(case.output_text)
    verdicts = [verifier.verify(claim, case.source_context) for claim in claims]
    score, counts = summarize(verdicts)
    return EvalResult(
        case=case,
        claims=verdicts,
        faithfulness_score=score,
        counts=counts,
        verifier_name=verifier.name,
    )

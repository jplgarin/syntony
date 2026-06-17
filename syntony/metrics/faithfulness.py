from __future__ import annotations

from collections.abc import Sequence

from ..schema import ClaimVerdict, Label


def label_counts(claims: Sequence[ClaimVerdict]) -> dict[str, int]:
    """Count claims per label, always returning all three keys.

    The three buckets are reported separately and never merged. "Contradicted"
    (the context refutes the claim) and "unsupported" (the context is silent)
    are distinct failure modes: the first usually means the model invented a
    fact the source disproves, the second that it went beyond the source. A
    consumer that wants a single error bucket can sum them, but syntony will not
    make that choice for them.
    """
    counts = {label.value: 0 for label in Label}
    for verdict in claims:
        counts[verdict.label.value] += 1
    return counts


def faithfulness_score(claims: Sequence[ClaimVerdict]) -> float:
    """Fraction of claims the source context actually supports.

    This is the scalar gate value: supported / total. Both contradicted and
    unsupported claims lower it, but only the per-label counts tell you which.

    An output that decomposes into zero claims is treated as faithful (score
    1.0): there is nothing in it the source fails to support. Callers that want
    to treat empty output as a failure should check the claim count themselves.
    """
    if not claims:
        return 1.0
    supported = sum(1 for v in claims if v.label is Label.SUPPORTED)
    return supported / len(claims)


def summarize(claims: Sequence[ClaimVerdict]) -> tuple[float, dict[str, int]]:
    """Convenience pairing of the scalar score and the per-label counts."""
    return faithfulness_score(claims), label_counts(claims)

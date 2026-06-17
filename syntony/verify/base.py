from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import ClaimVerdict


class Verifier(ABC):
    """Judges a single claim against a source context.

    Every verifier maps to the same three-label output and must attach the
    evidence behind its call (except for UNSUPPORTED, where the point is that
    no evidence exists). Subclasses range from a deterministic lexical baseline
    to a local NLI model; the report layer compares them precisely because they
    have different blind spots.
    """

    #: Stable identifier recorded on every verdict and piece of evidence.
    name: str = "verifier"

    @abstractmethod
    def verify(self, claim: str, context: str) -> ClaimVerdict:
        ...

from __future__ import annotations

from abc import ABC, abstractmethod


class Decomposer(ABC):
    """Splits an LLM output into atomic claims.

    A claim is the smallest unit we can hand to a verifier and get a meaningful
    verdict on. The quality of decomposition bounds the quality of everything
    downstream: merge two facts into one claim and a verdict can no longer tell
    you which half failed. The interface is kept trivial on purpose so smarter
    decomposers (clause-level, model-based) can drop in without touching the
    verification layer.
    """

    @abstractmethod
    def decompose(self, text: str) -> list[str]:
        ...

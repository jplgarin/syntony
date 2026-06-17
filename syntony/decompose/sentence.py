from __future__ import annotations

import re

from .base import Decomposer

# Sentence-final punctuation followed by whitespace and a capital/quote/digit.
# We deliberately avoid an NLP dependency here: a regex is fully deterministic,
# has no model to download, and is trivial to reason about when a split looks
# wrong. The cost is that it mishandles abbreviations, which we mitigate with a
# small protected-token pass below rather than a statistical model.
_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])      # a sentence-ending mark
    ["')\]]*        # optional closing quote/bracket attached to it
    \s+             # the gap before the next sentence
    (?=[A-Z0-9"'(\[])  # next sentence starts with a capital, digit, or quote
    """,
    re.VERBOSE,
)

# Common abbreviations whose trailing period must not end a sentence. Kept
# short and obvious; the goal is to avoid the most frequent bad splits, not to
# be exhaustive.
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "St.",
    "approx.",
    "cf.",
    "Fig.",
    "No.",
)

_PLACEHOLDER = "\x00"


class SentenceDecomposer(Decomposer):
    """Sentence-level decomposition.

    One sentence becomes one claim. This is a coarse but honest baseline: it
    over-merges compound sentences ("X is true and Y is false") into a single
    claim, which is a known limitation documented in the verifier tradeoffs.
    For most grounded-generation outputs, sentence granularity is enough to
    localise a faithfulness failure.
    """

    def __init__(self, min_chars: int = 1) -> None:
        # Filters out fragments left by stray punctuation or list bullets.
        self.min_chars = min_chars

    def decompose(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        protected = self._protect_abbreviations(text)
        parts = _SENTENCE_BOUNDARY.split(protected)

        claims: list[str] = []
        for part in parts:
            restored = part.replace(_PLACEHOLDER, ".").strip()
            if len(restored) >= self.min_chars:
                claims.append(restored)
        return claims

    @staticmethod
    def _protect_abbreviations(text: str) -> str:
        # Swap the period inside known abbreviations for a placeholder so the
        # boundary regex cannot fire on it, then swap back after splitting.
        for abbr in _ABBREVIATIONS:
            text = text.replace(abbr, abbr.replace(".", _PLACEHOLDER))
        return text

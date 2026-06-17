from __future__ import annotations

import re

from ..schema import ClaimVerdict, Evidence, Label
from .base import Verifier

# Words carrying no propositional content. Kept small on purpose: an aggressive
# stoplist would discard tokens that matter for grounding (numbers, names).
_STOPWORDS = frozenset(
    """
    a an the of to in on at by for with from into over under as is are was were
    be been being and or but if then this that these those it its their there
    here we you they he she them his her our your i me my mine ours yours
    """.split()
)

# Negation cues handled separately from content. We strip these out of the
# token overlap (so "not effective" still matches "effective") and use them
# only to decide polarity. "n't" is normalised to "not" before tokenising.
_NEGATIONS = frozenset(
    """
    not no never none neither nor without cannot cant lacks lacking absent
    fails failed fail unable
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,]\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("n't", " not"))


def _content_and_polarity(tokens: list[str]) -> tuple[list[str], int]:
    """Split a token stream into content tokens and a negation polarity.

    Polarity is the parity of negation cues: two negations cancel, which is
    crude but matches how double negation usually reads in practice.
    """
    content = [t for t in tokens if t not in _STOPWORDS and t not in _NEGATIONS]
    negations = sum(1 for t in tokens if t in _NEGATIONS)
    return content, negations % 2


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


class LexicalVerifier(Verifier):
    """Deterministic, fully offline grounding check.

    The model here is intentionally simple and transparent: a claim is
    *supported* when most of its content words (and ideally its word pairs)
    appear together in a single context sentence. The same machinery detects
    likely *contradiction*: when a claim lines up strongly with a context span
    but their negation polarities disagree, the context is saying the opposite
    of the claim. Everything else is *unsupported*.

    This is the reproducible baseline. It has no concept of paraphrase or
    synonymy, so it under-detects support that is phrased differently; that is
    the price of determinism and is exactly why the NLI verifier exists.
    """

    name = "lexical"

    def __init__(self, support_threshold: float = 0.6, unigram_weight: float = 0.7):
        if not 0.0 < support_threshold <= 1.0:
            raise ValueError("support_threshold must be in (0, 1]")
        if not 0.0 <= unigram_weight <= 1.0:
            raise ValueError("unigram_weight must be in [0, 1]")
        self.support_threshold = support_threshold
        self.unigram_weight = unigram_weight

    def verify(self, claim: str, context: str) -> ClaimVerdict:
        claim_tokens = _tokenize(claim)
        claim_content, claim_polarity = _content_and_polarity(claim_tokens)

        if not claim_content:
            # Nothing to ground (claim is all stopwords/punctuation). Treat as
            # unsupported rather than guess.
            return ClaimVerdict(
                claim=claim,
                label=Label.UNSUPPORTED,
                evidence=[],
                confidence=0.5,
                verifier=self.name,
            )

        best_score = 0.0
        best_sentence = ""
        best_polarity = 0
        for index, sentence in enumerate(self._sentences(context)):
            score, polarity = self._score(claim_content, claim_polarity, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_polarity = polarity

        if best_score < self.support_threshold:
            # The context never lines up well enough to affirm or refute. This
            # is silence, not refutation, so we record no evidence span.
            return ClaimVerdict(
                claim=claim,
                label=Label.UNSUPPORTED,
                evidence=[],
                confidence=round(1.0 - best_score, 4),
                verifier=self.name,
            )

        evidence = [
            Evidence(
                source_span=best_sentence,
                score=round(best_score, 4),
                verifier=self.name,
            )
        ]
        if best_polarity != claim_polarity:
            # Strong lexical overlap but opposite polarity: the span addresses
            # the claim and negates it.
            label = Label.CONTRADICTED
        else:
            label = Label.SUPPORTED

        return ClaimVerdict(
            claim=claim,
            label=label,
            evidence=evidence,
            confidence=round(best_score, 4),
            verifier=self.name,
        )

    def _score(
        self, claim_content: list[str], claim_polarity: int, sentence: str
    ) -> tuple[float, int]:
        ctx_tokens = _tokenize(sentence)
        ctx_content, ctx_polarity = _content_and_polarity(ctx_tokens)
        if not ctx_content:
            return 0.0, ctx_polarity

        claim_set = set(claim_content)
        ctx_set = set(ctx_content)
        unigram_cov = len(claim_set & ctx_set) / len(claim_set)

        claim_bigrams = _bigrams(claim_content)
        if claim_bigrams:
            bigram_cov = len(claim_bigrams & _bigrams(ctx_content)) / len(
                claim_bigrams
            )
        else:
            # Single-content-word claim: nothing to measure at bigram level, so
            # lean entirely on unigram coverage.
            bigram_cov = unigram_cov

        score = self.unigram_weight * unigram_cov + (
            1.0 - self.unigram_weight
        ) * bigram_cov
        return score, ctx_polarity

    @staticmethod
    def _sentences(context: str) -> list[str]:
        stripped = context.strip()
        if not stripped:
            return []
        return [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]

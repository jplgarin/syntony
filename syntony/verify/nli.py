from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..schema import ClaimVerdict, Evidence, Label
from .base import Verifier

if TYPE_CHECKING:
    # Only for type checkers; never imported at runtime so the core install
    # stays free of transformers/torch.
    from transformers import Pipeline


# Standard NLI label vocabulary varies by checkpoint (ENTAILMENT vs LABEL_0).
# We normalise by substring so the same code works across common MNLI models.
def _normalize_nli_label(raw: str) -> str | None:
    lowered = raw.lower()
    if "entail" in lowered:
        return "entailment"
    if "contradict" in lowered:
        return "contradiction"
    if "neutral" in lowered:
        return "neutral"
    return None


_NLI_TO_LABEL = {
    "entailment": Label.SUPPORTED,
    "contradiction": Label.CONTRADICTED,
    "neutral": Label.UNSUPPORTED,
}


class NLIError(RuntimeError):
    """Raised when the NLI backend cannot be loaded or used.

    Carries an actionable message. We never silently fall back to the lexical
    verifier: a caller who asked for NLI and got lexical results without
    knowing would draw false conclusions about their pipeline.
    """


class NLIVerifier(Verifier):
    """Local natural-language-inference verifier.

    Each context sentence is treated as a premise and the claim as the
    hypothesis. Entailment maps to supported, contradiction to contradicted,
    and neutral to unsupported. The verdict is the strongest signal across all
    sentences, and the sentence that produced it is recorded as evidence.

    The model runs locally on CPU and is loaded lazily on first use, so
    ``import syntony`` costs nothing and an offline lexical-only workflow never
    touches transformers. If the dependency or model is missing, this raises
    :class:`NLIError` rather than degrading silently.
    """

    name = "nli"

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        decision_threshold: float = 0.5,
        device: int = -1,
    ) -> None:
        if not 0.0 < decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in (0, 1]")
        self.model_name = model_name
        self.decision_threshold = decision_threshold
        self.device = device
        self._pipeline: Pipeline | None = None

    def _ensure_pipeline(self) -> Callable[..., Any]:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise NLIError(
                "The NLI verifier needs the 'transformers' and 'torch' "
                "packages. Install the optional extra with: "
                "pip install 'syntony[nli]'"
            ) from exc
        try:
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            raise NLIError(
                f"Could not load NLI model '{self.model_name}'. Check the "
                f"model name and that it is available locally or downloadable. "
                f"Underlying error: {exc}"
            ) from exc
        return self._pipeline

    def verify(self, claim: str, context: str) -> ClaimVerdict:
        sentences = self._sentences(context)
        if not sentences:
            return ClaimVerdict(
                claim=claim,
                label=Label.UNSUPPORTED,
                evidence=[],
                confidence=0.5,
                verifier=self.name,
            )

        pipe = self._ensure_pipeline()

        best_entail = (0.0, "")
        best_contra = (0.0, "")
        for sentence in sentences:
            scores = self._infer(pipe, premise=sentence, hypothesis=claim)
            entail = scores.get("entailment", 0.0)
            contra = scores.get("contradiction", 0.0)
            if entail > best_entail[0]:
                best_entail = (entail, sentence)
            if contra > best_contra[0]:
                best_contra = (contra, sentence)

        return self._decide(claim, best_entail, best_contra)

    def _decide(
        self,
        claim: str,
        best_entail: tuple[float, str],
        best_contra: tuple[float, str],
    ) -> ClaimVerdict:
        entail_score, entail_span = best_entail
        contra_score, contra_span = best_contra

        if entail_score >= contra_score and entail_score >= self.decision_threshold:
            return ClaimVerdict(
                claim=claim,
                label=Label.SUPPORTED,
                evidence=[
                    Evidence(
                        source_span=entail_span,
                        score=round(entail_score, 4),
                        verifier=self.name,
                    )
                ],
                confidence=round(entail_score, 4),
                verifier=self.name,
            )
        if contra_score > entail_score and contra_score >= self.decision_threshold:
            return ClaimVerdict(
                claim=claim,
                label=Label.CONTRADICTED,
                evidence=[
                    Evidence(
                        source_span=contra_span,
                        score=round(contra_score, 4),
                        verifier=self.name,
                    )
                ],
                confidence=round(contra_score, 4),
                verifier=self.name,
            )

        # Neutral dominates, or neither signal clears the threshold.
        confidence = round(1.0 - max(entail_score, contra_score), 4)
        return ClaimVerdict(
            claim=claim,
            label=Label.UNSUPPORTED,
            evidence=[],
            confidence=confidence,
            verifier=self.name,
        )

    @staticmethod
    def _infer(
        pipe: Callable[..., Any], premise: str, hypothesis: str
    ) -> dict[str, float]:
        # transformers text-classification accepts a {text, text_pair} dict for
        # sentence-pair models. With top_k=None it returns all class scores.
        raw = pipe({"text": premise, "text_pair": hypothesis})
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        scores: dict[str, float] = {}
        for entry in raw:
            mapped = _normalize_nli_label(entry["label"])
            if mapped is not None:
                scores[mapped] = float(entry["score"])
        return scores

    @staticmethod
    def _sentences(context: str) -> list[str]:
        from ..decompose.sentence import SentenceDecomposer

        return SentenceDecomposer().decompose(context)

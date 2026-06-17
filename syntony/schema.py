"""Core data model.

Everything syntony produces is built from these types. The central design
constraint is auditability: a verdict is worthless in a regulated setting if
you cannot point at the exact span of source text that produced it, so
``ClaimVerdict`` always carries the ``Evidence`` that justified its label.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Label(str, Enum):
    """The three faithfulness outcomes for a single claim.

    The split between CONTRADICTED and UNSUPPORTED is deliberate and load
    bearing. A claim the context actively refutes is a different (and usually
    more serious) failure than a claim the context simply never addresses.
    Collapsing them hides that distinction, so we never do.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"

    def __str__(self) -> str:
        return self.value


class Evidence(BaseModel):
    """A concrete span of source text behind a verdict.

    ``location`` is intentionally loose (a character offset, a sentence index,
    a section name, whatever the verifier can offer) because different
    verifiers locate evidence differently and we would rather record what we
    have than force a single scheme.
    """

    model_config = ConfigDict(frozen=True)

    source_span: str
    location: str | int | None = None
    score: float = Field(ge=0.0, le=1.0)
    verifier: str


class ClaimVerdict(BaseModel):
    """The verdict for one atomic claim, with the evidence that produced it."""

    model_config = ConfigDict(use_enum_values=False)

    claim: str
    label: Label
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    verifier: str

    @model_validator(mode="after")
    def _supported_or_contradicted_needs_evidence(self) -> ClaimVerdict:
        # A positive or negative call has to be grounded in something. Only
        # UNSUPPORTED is allowed to stand on the absence of evidence.
        if self.label is not Label.UNSUPPORTED and not self.evidence:
            raise ValueError(
                f"label '{self.label}' requires at least one Evidence span"
            )
        return self


class EvalCase(BaseModel):
    """One verification task: an output to check against its source context."""

    output_text: str
    source_context: str
    metadata: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    """The full, self-contained result of verifying an ``EvalCase``.

    Holds the original case so the result can be archived on its own and still
    be reproducible and inspectable later.
    """

    case: EvalCase
    claims: list[ClaimVerdict]
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    counts: dict[str, int]
    verifier_name: str

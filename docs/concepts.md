# Concepts: claims, verdicts, evidence

syntony turns a vague question ("is this LLM output faithful to its source?")
into a set of small, checkable questions with auditable answers.

## Claim

A claim is the smallest unit of an output that can be judged true or false
against the source. An LLM answer is rarely atomic: it bundles several
assertions into one paragraph. syntony first decomposes the output into claims
so that a failure can be localised to the exact assertion that failed, instead
of marking the whole answer "unfaithful".

The default `SentenceDecomposer` treats each sentence as one claim. This is a
deliberate, transparent baseline. It over-merges compound sentences (for
example "X holds and Y does not"), which is the main reason the decomposer is
an interface: a finer decomposer can be dropped in without touching anything
downstream.

## Verdict

Each claim is verified against the source context and receives one of three
labels:

| Label          | Meaning                                  |
| -------------- | ---------------------------------------- |
| `supported`    | The context entails the claim.           |
| `contradicted` | The context refutes the claim.           |
| `unsupported`  | The context neither entails nor refutes. |

### Why three labels, not two

The interesting distinction is between the two failure modes.

- **Contradicted** means the model asserted something the source actively
  disproves. In a clinical or financial setting this is the dangerous case: the
  ground truth was available and the output disagrees with it.
- **Unsupported** means the model went beyond the source. The source is silent.
  This may be a hallucination, or it may be a reasonable inference, or the
  retrieval simply missed the relevant passage.

A binary faithful/unfaithful score hides which of these happened. They call for
different responses (fix the model versus fix the retrieval), so syntony keeps
them separate everywhere except in the single scalar score, and even there the
per-label counts travel alongside.

## Evidence

A verdict without its justification is not auditable, and an unauditable
verdict is useless in a regulated environment. So every `supported` or
`contradicted` verdict carries at least one `Evidence` span: the concrete piece
of source text that entailed or refuted the claim, with a score and the name of
the verifier that produced it. This is enforced in the schema, not left to
convention.

`unsupported` is the one label allowed to carry no evidence, because the
absence of any relevant span is precisely what "unsupported" means.

## Faithfulness score

The scalar score is `supported / total_claims`: the fraction of the output that
the source actually backs. Both contradicted and unsupported claims lower it.
The score is the number you gate on; the counts are the number you debug with.

An output that decomposes into zero claims scores 1.0 by convention (there is
nothing in it the source fails to support). If empty output should count as a
failure in your pipeline, check the claim count yourself.

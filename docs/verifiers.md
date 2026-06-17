# Verifiers

A verifier answers one question: given a claim and a source context, is the
claim supported, contradicted, or unsupported? syntony ships two, with
different cost and different blind spots.

## Lexical verifier (`LexicalVerifier`)

Deterministic, fully offline, no model. This is the reproducible baseline.

### How it works

1. The claim and each context sentence are tokenised. Stopwords and negation
   cues are stripped out so the comparison is about propositional content, not
   grammar. Contractions like `doesn't` are normalised to `does not` first.
2. For each context sentence, it computes how much of the claim's content is
   present: unigram coverage (which content words appear) blended with bigram
   coverage (which word pairs appear). The bigram term rewards genuine phrase
   matches and penalises claims that merely share scattered words.
3. The best-scoring context sentence becomes the candidate evidence.
4. Negation polarity is compared separately. If the claim and the matched
   sentence line up strongly but disagree on negation (one says "effective",
   the other "not effective"), the verdict is `contradicted`. If they agree, it
   is `supported`. If no sentence clears the support threshold, the verdict is
   `unsupported`.

### Strengths

- Fully deterministic and reproducible: same input, same verdict, forever.
- No network, no download, no GPU. Runs anywhere Python runs.
- Transparent: you can read the score and see exactly why a verdict was made.

### Limits

- No notion of synonymy or paraphrase. "The medication lowered deaths" will not
  match "The drug reduced mortality", so it under-detects support that is
  reworded.
- It only catches contradiction expressed through explicit negation cues. It
  cannot tell that "increased" contradicts "reduced" (antonyms), and will
  usually call such a claim `unsupported` rather than `contradicted`.

Tune `support_threshold` (default 0.6) to trade recall for precision, and
`unigram_weight` (default 0.7) to shift emphasis between single words and
phrases.

## NLI verifier (`NLIVerifier`)

A local natural-language-inference model. Install with
`pip install 'syntony[nli]'`.

### How it works

Each context sentence is the premise and the claim is the hypothesis. The model
returns probabilities for entailment, neutral, and contradiction, which map
directly to supported, unsupported, and contradicted. The verdict is the
strongest signal across all sentences, and the sentence that produced it is
recorded as the evidence span.

The model loads lazily on first `verify` call, so importing syntony stays cheap
and an offline lexical workflow never pulls in transformers or torch. The model
name is configurable (default `facebook/bart-large-mnli`) and it runs on CPU by
default.

If transformers or the model is unavailable, the verifier raises `NLIError`
with the exact fix. It never silently falls back to the lexical verifier: a
caller who asked for NLI and unknowingly got lexical results would draw wrong
conclusions about their system.

### Strengths

- Handles paraphrase and semantic entailment that the lexical verifier misses.
- Detects contradiction that is not signalled by an explicit negation word.

### Limits

- Not deterministic across model versions or hardware in the way the lexical
  verifier is. Pin the model name for reproducibility.
- Heavier: needs the model downloaded once and is slower per claim.
- The verdict is only as calibrated as the underlying NLI model.

## Choosing, and using both

Use the lexical verifier as the always-available, reproducible gate, including
in air-gapped or regulated environments. Use the NLI verifier when you need to
catch reworded support and subtler contradiction.

The strongest setup runs both and inspects where they disagree. That is what
the agreement report is for: disagreement between a deterministic lexical check
and a semantic model is a reliable signal that a claim deserves a human look.
See `syntony/agreement/report.py` and the `--agreement` CLI flag.

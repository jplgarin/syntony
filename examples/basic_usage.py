"""Run the lexical verifier over one case and print each verdict with evidence.

This uses only the core install (no transformers). Run it with:

    python examples/basic_usage.py
"""

from syntony import EvalCase, LexicalVerifier, evaluate

case = EvalCase(
    output_text=(
        "The study enrolled 200 patients. "
        "The treatment was effective for chronic pain. "
        "The drug was approved in 2019."
    ),
    source_context=(
        "A trial enrolled 200 patients across four sites. "
        "The treatment was not effective for chronic pain. "
        "The drug reduced mortality by 30 percent."
    ),
)

result = evaluate(case, LexicalVerifier())

print(f"faithfulness: {result.faithfulness_score:.2f}")
print(f"counts: {result.counts}\n")

for verdict in result.claims:
    print(f"[{verdict.label}] {verdict.claim}")
    for evidence in verdict.evidence:
        print(f"    evidence ({evidence.score}): {evidence.source_span}")
    if not verdict.evidence:
        print("    no supporting or contradicting span in the context")

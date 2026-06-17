from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__, evaluate
from .agreement import agreement_report
from .schema import EvalCase, EvalResult
from .verify import LexicalVerifier, NLIError, NLIVerifier, Verifier

# Exit codes are part of the contract for CI use, so they are fixed and small.
EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1  # ran fine, but faithfulness fell under --fail-under
EXIT_USAGE = 2  # bad arguments or input the user can fix
EXIT_RUNTIME = 3  # a backend failed (e.g. NLI model could not load)


def _build_verifier(name: str, model_name: str | None) -> Verifier:
    if name == "lexical":
        return LexicalVerifier()
    if name == "nli":
        return NLIVerifier(model_name=model_name) if model_name else NLIVerifier()
    raise ValueError(f"unknown verifier '{name}'")


def _load_cases(path: str) -> list[EvalCase]:
    """Accept either a single EvalCase object or a JSON array of them."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records = data if isinstance(data, list) else [data]
    return [EvalCase.model_validate(record) for record in records]


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    return {
        "verifier": result.verifier_name,
        "faithfulness_score": round(result.faithfulness_score, 4),
        "counts": result.counts,
        "claims": [
            {
                "claim": c.claim,
                "label": c.label.value,
                "confidence": c.confidence,
                "evidence": [
                    {
                        "source_span": e.source_span,
                        "location": e.location,
                        "score": e.score,
                        "verifier": e.verifier,
                    }
                    for e in c.evidence
                ],
            }
            for c in result.claims
        ],
    }


def _print_human(result: EvalResult, stream: Any) -> None:
    print(f"verifier: {result.verifier_name}", file=stream)
    print(
        f"faithfulness: {result.faithfulness_score:.4f}  "
        f"(supported={result.counts['supported']} "
        f"contradicted={result.counts['contradicted']} "
        f"unsupported={result.counts['unsupported']})",
        file=stream,
    )
    for c in result.claims:
        print(f"  [{c.label.value}] {c.claim}  (conf={c.confidence})", file=stream)
        for e in c.evidence:
            print(f"      evidence: {e.source_span!r} (score={e.score})", file=stream)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syntony",
        description="Offline, traceable grounding and faithfulness verifier.",
    )
    parser.add_argument("input", help="Path to a JSON EvalCase or array of EvalCases.")
    parser.add_argument(
        "--verifier",
        choices=["lexical", "nli"],
        default="lexical",
        help="Verifier backend (default: lexical, fully offline).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the NLI model name (only used with --verifier nli).",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Exit non-zero if any case scores below THRESHOLD (0..1). "
        "Use as a grounding gate in CI.",
    )
    parser.add_argument(
        "--agreement",
        action="store_true",
        help="Also run the lexical and NLI verifiers and report their agreement.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    parser.add_argument("--version", action="version", version=f"syntony {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fail_under is not None and not 0.0 <= args.fail_under <= 1.0:
        parser.error("--fail-under must be between 0 and 1")

    try:
        cases = _load_cases(args.input)
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return EXIT_USAGE
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: could not parse input: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        verifier = _build_verifier(args.verifier, args.model_name)
        results = [evaluate(case, verifier) for case in cases]
        agreement_blocks = _maybe_agreement(args, cases)
    except NLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    _emit(args, results, agreement_blocks)

    if args.fail_under is not None:
        worst = min(r.faithfulness_score for r in results) if results else 1.0
        if worst < args.fail_under:
            print(
                f"FAIL: faithfulness {worst:.4f} < threshold {args.fail_under:.4f}",
                file=sys.stderr,
            )
            return EXIT_BELOW_THRESHOLD

    return EXIT_OK


def _maybe_agreement(args: argparse.Namespace, cases: Sequence[EvalCase]) -> list[Any]:
    if not args.agreement:
        return []
    verifiers: list[Verifier] = [
        LexicalVerifier(),
        NLIVerifier(model_name=args.model_name) if args.model_name else NLIVerifier(),
    ]
    return [agreement_report(case, verifiers) for case in cases]


def _emit(
    args: argparse.Namespace,
    results: Sequence[EvalResult],
    agreement_blocks: Sequence[Any],
) -> None:
    if args.as_json:
        payload: dict[str, Any] = {
            "results": [_result_to_dict(r) for r in results]
        }
        if agreement_blocks:
            payload["agreement"] = [
                json.loads(block.model_dump_json()) for block in agreement_blocks
            ]
        print(json.dumps(payload, indent=2))
        return

    for index, result in enumerate(results):
        if len(results) > 1:
            print(f"=== case {index} ===")
        _print_human(result, sys.stdout)
        if agreement_blocks:
            block = agreement_blocks[index]
            print(f"  agreement rate: {block.overall_agreement_rate}")
            for pk in block.pairwise_kappa:
                print(f"  kappa({pk.verifier_a},{pk.verifier_b}): {pk.kappa}")
            for d in block.disagreements:
                print(f"  DISAGREE: {d.claim}  {d.labels}")


if __name__ == "__main__":
    raise SystemExit(main())

# CI gating with `--fail-under`

syntony is built to run as a grounding gate in a pipeline: decompose the
output, verify every claim, compute the faithfulness score, and fail the build
if the score drops below a threshold.

## The command

```bash
syntony cases.json --fail-under 0.8
```

The input is a single `EvalCase` or a JSON array of them. For a batch, the gate
uses the worst (lowest) score across all cases, so one bad case fails the run.

## Exit codes

The exit codes are fixed and part of the contract, so a pipeline can branch on
them:

| Code | Meaning                                                          |
| ---- | --------------------------------------------------------------- |
| 0    | Ran successfully and met the threshold (or no threshold given). |
| 1    | Ran successfully but faithfulness fell below `--fail-under`.    |
| 2    | Usage error: bad arguments, missing file, or unparseable JSON.  |
| 3    | Runtime error: a backend failed (for example the NLI model could not load). |

Distinguishing 1 from 2 and 3 matters: a below-threshold result (1) is a real
finding about your output, while 2 and 3 mean the gate did not actually run and
should be treated as an infrastructure failure, not a passing build.

## In GitHub Actions

```yaml
- name: Grounding gate
  run: syntony eval_cases.json --fail-under 0.8
```

The default lexical verifier needs no model and no network, so this step works
in any runner with no extra setup. The step fails the job on exit code 1.

## Choosing a threshold

Start by running without `--fail-under` to see the score distribution on a
representative set of outputs, then set the threshold just below your accepted
baseline so regressions trip it. Inspect the per-label counts: a score that
drops because of `contradicted` claims is more urgent than the same drop caused
by `unsupported` claims, since contradiction means the output disagrees with a
source you already had.

## Machine-readable output

Add `--json` to capture full per-claim verdicts and evidence for archiving or
further processing. The exit-code behaviour is unchanged, so you can both record
the detail and gate on the score in one step.

```bash
syntony eval_cases.json --fail-under 0.8 --json > grounding_report.json
```

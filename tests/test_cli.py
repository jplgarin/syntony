import json
import sys

import pytest

from syntony import cli
from syntony.schema import ClaimVerdict, Evidence, Label
from syntony.verify import Verifier

GROUNDED = {
    "output_text": "The trial enrolled 200 patients.",
    "source_context": "The trial enrolled 200 patients in total.",
    "metadata": {},
}
UNGROUNDED = {
    "output_text": "The company was founded on Mars in 1850.",
    "source_context": "The trial enrolled 200 patients in total.",
    "metadata": {},
}


def _write(tmp_path, payload):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_lexical_run_exit_ok(tmp_path, capsys):
    code = cli.main([_write(tmp_path, GROUNDED), "--verifier", "lexical"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "faithfulness" in out
    assert "supported" in out


def test_fail_under_triggers_exit_one(tmp_path, capsys):
    code = cli.main([_write(tmp_path, UNGROUNDED), "--fail-under", "0.9"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_BELOW_THRESHOLD
    assert "FAIL" in err


def test_fail_under_passes(tmp_path):
    code = cli.main([_write(tmp_path, GROUNDED), "--fail-under", "0.9"])
    assert code == cli.EXIT_OK


def test_json_output_is_parseable(tmp_path, capsys):
    code = cli.main([_write(tmp_path, GROUNDED), "--json"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["results"][0]["counts"]["supported"] == 1
    assert payload["results"][0]["claims"][0]["evidence"]


def test_batch_input(tmp_path, capsys):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps([GROUNDED, UNGROUNDED]), encoding="utf-8")
    assert cli.main([str(path), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["results"]) == 2


def test_fail_under_uses_worst_case(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps([GROUNDED, UNGROUNDED]), encoding="utf-8")
    assert cli.main([str(path), "--fail-under", "0.5"]) == cli.EXIT_BELOW_THRESHOLD


def test_file_not_found(tmp_path, capsys):
    code = cli.main([str(tmp_path / "missing.json")])
    assert code == cli.EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_bad_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not valid", encoding="utf-8")
    code = cli.main([str(path)])
    assert code == cli.EXIT_USAGE


def test_invalid_fail_under_value(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([_write(tmp_path, GROUNDED), "--fail-under", "2.0"])
    assert exc.value.code == cli.EXIT_USAGE


def test_nli_missing_dependency_exit_runtime(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    code = cli.main([_write(tmp_path, GROUNDED), "--verifier", "nli"])
    assert code == cli.EXIT_RUNTIME
    assert "syntony[nli]" in capsys.readouterr().err


class _ScriptedNLI(Verifier):
    name = "nli"

    def __init__(self, *args, **kwargs):
        pass

    def verify(self, claim, context):
        return ClaimVerdict(
            claim=claim,
            label=Label.SUPPORTED,
            evidence=[Evidence(source_span=context[:10], score=0.9, verifier="nli")],
            confidence=0.9,
            verifier="nli",
        )


def test_agreement_flag(tmp_path, capsys, monkeypatch):
    # Avoid loading a real model: swap NLIVerifier for a scripted stand-in.
    monkeypatch.setattr(cli, "NLIVerifier", _ScriptedNLI)
    code = cli.main([_write(tmp_path, GROUNDED), "--agreement"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "agreement rate" in out


def test_agreement_flag_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "NLIVerifier", _ScriptedNLI)
    assert cli.main([_write(tmp_path, GROUNDED), "--agreement", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "agreement" in payload
    assert payload["agreement"][0]["verifiers"] == ["lexical", "nli"]

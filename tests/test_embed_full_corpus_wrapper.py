"""Contract tests for the resumable full-corpus shell wrapper."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
WRAPPER = REPO_ROOT / "scripts" / "embed_full_corpus.sh"


def _fake_python(tmp_path: Path) -> tuple[Path, Path, Path]:
    calls = tmp_path / "calls.jsonl"
    state = tmp_path / "embedded"
    executable = tmp_path / "fake-python"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['FAKE_CALLS'], 'a', encoding='utf-8') as out:\n"
        "    out.write(json.dumps(args) + '\\n')\n"
        "name = pathlib.Path(args[0]).name\n"
        "if name == 'embed_status.py':\n"
        "    if '--source-summary' in args:\n"
        "        if '--corpus-db' in args:\n"
        "            print('source_kind=tvi_canonical_db generation_id=tvi-generation-v1:sha256:' + '1' * 64 + ' logical_digest_sha256=' + '2' * 64 + ' input_keys=base,delta')\n"
        "        else:\n"
        "            print('source_kind=generic_jsonl path=/tmp/generic.jsonl sha256=' + '3' * 64)\n"
        "    elif pathlib.Path(os.environ['FAKE_STATE']).exists()"
        " and not os.environ.get('FAKE_INCOMPLETE'):\n"
        "        print('1 0 0 1')\n"
        "    else:\n"
        "        print('0 0 1 1')\n"
        "elif name == 'embed_corpus.py':\n"
        "    pathlib.Path(os.environ['FAKE_STATE']).touch()\n"
        "else:\n"
        "    raise SystemExit('unexpected child: ' + name)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    return executable, calls, state


def _run_wrapper(
    tmp_path: Path, *, expected_returncode: int = 0, **source_env: str
) -> tuple[list[list[str]], str, str, int]:
    fake_python, calls_path, state = _fake_python(tmp_path)
    log = tmp_path / "embed.log"
    checkpoint = tmp_path / "checkpoint.jsonl"
    env = os.environ.copy()
    for key in (
        "CORPUS",
        "CORPUS_DB",
        "INPUT_KEYS",
        "CANONICAL_ROOT",
        "TVI_CANONICAL_CORPUS_EXECUTABLE",
    ):
        env.pop(key, None)
    env.update(
        {
            "PY": str(fake_python),
            "FAKE_CALLS": str(calls_path),
            "FAKE_STATE": str(state),
            "CKPT": str(checkpoint),
            "LOG": str(log),
            "SLEEP_SECONDS": "0",
            **source_env,
        }
    )
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == expected_returncode, completed.stderr + "\n" + completed.stdout
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    return calls, log.read_text(encoding="utf-8"), completed.stdout, completed.returncode


def _calls_for(calls: list[list[str]], script_name: str) -> list[list[str]]:
    return [call for call in calls if Path(call[0]).name == script_name]


def test_wrapper_forwards_canonical_source_to_embed_and_status(tmp_path):
    calls, log, _, _ = _run_wrapper(
        tmp_path,
        CORPUS_DB="/srv/tvi/output/canonical/corpus.db",
        INPUT_KEYS="base delta",
        CANONICAL_ROOT="/srv/tvi/output/canonical",
    )

    for script_name in ("embed_status.py", "embed_corpus.py"):
        child_calls = _calls_for(calls, script_name)
        assert child_calls
        for call in child_calls:
            assert call.count("--corpus-db") == 1
            assert call[call.index("--corpus-db") + 1] == "/srv/tvi/output/canonical/corpus.db"
            assert call.count("--input-key") == 2
            assert "base" in call and "delta" in call
            assert call[call.index("--canonical-root") + 1] == "/srv/tvi/output/canonical"

    assert "generation_id=tvi-generation-v1:sha256:" in log
    assert "logical_digest_sha256=" in log
    assert "input_keys=base,delta" in log


def test_wrapper_retains_explicit_generic_corpus_mode(tmp_path):
    corpus = tmp_path / "generic.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"alpha"}\n', encoding="utf-8")
    calls, log, _, _ = _run_wrapper(tmp_path, CORPUS=str(corpus))

    for script_name in ("embed_status.py", "embed_corpus.py"):
        child_calls = _calls_for(calls, script_name)
        assert child_calls
        for call in child_calls:
            assert str(corpus) in call
            assert "--corpus-db" not in call
            assert "--input-key" not in call

    assert "source_kind=generic_jsonl" in log
    assert "generation_id=" not in log


def test_wrapper_incomplete_status_stalls_nonzero_without_done(tmp_path):
    corpus = tmp_path / "generic.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"alpha"}\n', encoding="utf-8")

    _calls, log, stdout, returncode = _run_wrapper(
        tmp_path,
        expected_returncode=1,
        CORPUS=str(corpus),
        FAKE_INCOMPLETE="1",
    )

    assert returncode == 1
    assert "[wrapper] DONE:" not in log
    assert "[wrapper] DONE:" not in stdout
    assert "[wrapper] STOP: no new successes across 3 passes" in log

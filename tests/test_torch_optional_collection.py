"""Regression guard for issue #62: suite collects clean without torch.

``tests/test_nv_embed_count_tokens.py`` imports ``NVEmbedAdapter``, whose
module (``nv_embed_adapter.py``) does ``import torch`` at module scope --
required at runtime for the GPU backend, but torch lives behind the optional
``gpu`` extra (pyproject.toml), not the base ``dev`` install. Before the fix,
collecting the suite in a torch-less environment raised a hard
``ModuleNotFoundError`` during collection and aborted the whole run. The fix
guards the import with ``pytest.importorskip`` so the module is skipped with
a documented reason instead.

This test simulates a torch-less environment via ``sys.modules`` (setting
``torch`` to ``None``, the standard trick for forcing ``ImportError`` on
subsequent imports -- see PEP 328 / importlib docs) and asserts collection of
the guarded module no longer errors.
"""

from __future__ import annotations

import subprocess
import sys


def test_collection_does_not_error_without_torch():
    """`pytest --collect-only` must not error when torch is unimportable.

    Runs in a subprocess so the parent test session's already-imported
    ``semantic_kinematics.embeddings.nv_embed_adapter`` (if any) can't mask
    the missing-torch path; ``sys.modules["torch"] = None`` forces every
    ``import torch`` in the subprocess to raise ImportError, standing in for
    a real environment where the `gpu` extra was never installed.
    """
    probe = (
        "import sys; sys.modules['torch'] = None; "
        "import pytest; "
        "sys.exit(pytest.main(["
        "'--collect-only', '-q', 'tests/test_nv_embed_count_tokens.py'"
        "]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Before the fix: returncode 2 ("Interrupted: 1 error during collection").
    # After the fix: returncode 5 ("no tests collected") because the module's
    # sole content becomes a skip -- collection itself raises nothing.
    assert "ModuleNotFoundError" not in result.stdout, result.stdout
    assert "error" not in result.stdout.lower(), result.stdout
    assert "1 skipped" in result.stdout, result.stdout
    assert result.returncode == 5, (result.returncode, result.stdout, result.stderr)

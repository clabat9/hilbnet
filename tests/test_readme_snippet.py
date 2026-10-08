"""The usage snippet in README.md runs and does what its comments say."""
from __future__ import annotations

import re
from pathlib import Path

import torch

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_usage_snippet_runs():
    text = README.read_text(encoding="utf-8")
    match = re.search(r"<!-- usage-snippet -->\s*```python\n(.*?)\n```", text, re.S)
    assert match, "usage snippet not found in README.md"
    namespace: dict = {}
    exec(compile(match.group(1), "README.md", "exec"), namespace)
    assert tuple(namespace["y"].shape) == (8, 4, 12)
    P = namespace["P"].detach()
    assert tuple(P.shape) == (4, 12, 12)
    eye = torch.eye(12).expand(4, 12, 12)
    assert torch.allclose(P.transpose(1, 2) @ P, eye, atol=1e-5)

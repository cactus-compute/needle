from __future__ import annotations

from pathlib import Path

import tomllib


def test_training_dependencies_are_not_required_for_runtime() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    runtime = set(project["project"]["dependencies"])
    training = set(project["project"]["optional-dependencies"]["train"])
    extras = project["project"]["optional-dependencies"]

    assert runtime.isdisjoint(training)
    assert extras["gpu"]
    assert extras["metal"]
    assert not set(extras["gpu"]).intersection(training)
    assert not set(extras["metal"]).intersection(training)

    requirements = {
        line.strip()
        for line in (root / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    train_requirements = {
        line.strip()
        for line in (root / "requirements-train.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-r ")
    }
    assert requirements == runtime
    assert train_requirements == training
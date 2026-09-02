"""Exercise the release workflow's Git operations against a local repository."""
from pathlib import Path
import subprocess
import textwrap

import pytest


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/release.yaml"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def run_block(repo, text, **env):
    import os
    subprocess.run(["bash", "-eu", "-c", textwrap.dedent(text)], cwd=repo,
                   env={**os.environ, **env}, check=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Release test")
    git(tmp_path, "config", "user.email", "release@example.invalid")
    (tmp_path / "needle").mkdir()
    (tmp_path / "pyproject.toml").write_text('version = "2.0.8"\n')
    (tmp_path / "needle/__init__.py").write_text('__version__ = "2.0.8"\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Initial source")
    return tmp_path


def version_output(repo):
    workflow = WORKFLOW.read_text()
    block = workflow.split("      - id: version\n        run: |\n", 1)[1]
    block = block.split("      - if:", 1)[0]
    output = repo / "output"
    output.write_text("")
    run_block(repo, block, GITHUB_OUTPUT=str(output))
    return output.read_text().strip()


def test_existing_release_tag_skips_unchanged_source(repo):
    git(repo, "tag", "v2.0.11")
    assert version_output(repo) == "skip=true"


def test_release_tag_contains_versioned_source_and_skips_unchanged_main(repo):
    original = git(repo, "rev-parse", "HEAD")
    (repo / "pyproject.toml").write_text('version = "2.0.12"\n')
    (repo / "needle/__init__.py").write_text('__version__ = "2.0.12"\n')
    workflow = WORKFLOW.read_text()
    block = workflow.rsplit("        run: |\n", 1)[1]
    # The test uses a local tag and never pushes to a remote.
    block = block.split("          git push", 1)[0]
    block = block.replace("${{ steps.version.outputs.next }}", "2.0.12")
    run_block(repo, block)
    assert git(repo, "show", "v2.0.12:pyproject.toml") == 'version = "2.0.12"'
    assert git(repo, "show", "v2.0.12:needle/__init__.py") == '__version__ = "2.0.12"'
    assert git(repo, "rev-parse", "v2.0.12^") == original
    assert version_output(repo) == "skip=true"
    git(repo, "checkout", "--detach", original)
    assert version_output(repo) == "skip=true"

    (repo / "feature.py").write_text("feature = True\n")
    git(repo, "add", "feature.py")
    git(repo, "commit", "-m", "Add feature")
    assert version_output(repo) == "next=2.0.13"

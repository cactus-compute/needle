from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_installation_guide_has_cpu_setup_and_offline_flow():
    text = _read("doc/installation.md")
    for required in (
        "uv venv",
        'uv pip install -e ".[train,test]"',
        "needle fetch",
        "HF_HUB_OFFLINE=1",
    ):
        assert required in text


def test_inference_guide_has_cli_first_and_python_api_examples():
    text = _read("doc/inference.md")
    assert "needle run" in text
    assert "--max-len 16" in text
    assert "Needle" in text and "max_new_tokens=16" in text
    assert "@needle.tool" in text
    assert "function_calls" in text
    assert "needle fetch" in text
    assert "退出码" in text


def test_readme_links_to_onboarding_guides():
    text = _read("README.md")
    assert "doc/installation.md" in text
    assert "doc/inference.md" in text


def test_docs_do_not_require_exact_generated_text():
    text = _read("doc/inference.md")
    assert "不要把示例文本作为快照断言" in text
    assert "assert.*输出" not in text

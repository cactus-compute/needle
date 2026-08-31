# 安装与运行环境

这份指南是 Needle 的 CPU-first 上手路径。它使用仓库源码的 editable 安装，
并把训练和测试依赖一起装好，方便后续继续阅读微调教程。Phase 1 只验收
CPU 推理；NVIDIA CUDA 和 Apple Metal 的安装与验证留作后续 TODO。

## 前置条件

- CPython 3.9 或更高版本（项目声明为 `>=3.9`）。
- 可以访问 Hugging Face 的网络，用于首次下载原生引擎和模型资产。
- `uv` 0.4 或更高版本。可从 <https://docs.astral.sh/uv/getting-started/installation/>
  安装，先用 `uv --version` 确认命令可用。
- Linux/macOS/Windows 上的普通 CPU 环境；本阶段不需要 CUDA、Metal 或编译器。

## 用 uv 创建环境

在仓库根目录执行：

```sh
uv venv
```

这会在当前目录创建 `.venv`。激活它（每个新 shell 都需要重新激活）：

```sh
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

# Windows cmd.exe
# .venv\\Scripts\\activate.bat
```

确认 Python 版本在支持范围内，然后以 editable 模式安装源码及完整的训练、
测试 extras：

```sh
python --version
uv pip install -e ".[train,test]"
uv pip check
python -c "import needle; print(needle.__version__)"
```

`-e` 表示对当前 checkout 的修改会立即反映到环境中；`train` extra 提供
JAX、Flax、Optax、NumPy 和 SentencePiece，用于参考模型、LoRA 微调和导出；
`test` extra 提供 pytest 与 Pydantic，用于运行测试和结构化工具示例。运行
`uv pip check` 应输出 `No broken requirements found.`（不同 uv 版本可能有
轻微格式差异），最后一条命令应打印当前包版本（例如 `2.0.8`）。

> 如果 shell 找不到 `python` 或 `needle`，通常是尚未激活 `.venv`。重新执行
> 激活命令，或直接使用 `.venv/bin/python` 和 `.venv/bin/needle`。

### 可选后端（后续阶段）

本阶段不安装或验证加速后端。以后在 NVIDIA CUDA 12 机器上可研究
`uv pip install -e ".[train,gpu]"`，Apple Silicon 可研究
`uv pip install -e ".[train,metal]"`；两者都不属于当前 CPU 验收结果，具体
版本矩阵会在后续文档中单独确认。

## 关闭匿名遥测（可选）

默认只记录函数名、包版本和操作系统，不发送 prompt、输出或训练数据。若设备
策略禁止遥测，在激活环境后设置以下任一变量，再运行命令即可：

```sh
export NEEDLE_TELEMETRY=0
# 或
export DO_NOT_TRACK=1
```

Windows PowerShell 对应 `\$env:NEEDLE_TELEMETRY = "0"`。

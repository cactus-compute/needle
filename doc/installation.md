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

## 认识模型资产

安装 Python 包不会把所有训练文件放进仓库。先区分这些文件的职责：

| 资产 | 用途 | Phase 1 是否需要 |
| --- | --- | --- |
| 原生引擎 `libneedle.so`（macOS 为 `.dylib`，Windows 为 `.dll`） | 针对当前 CPU/平台的推理运行时；包含发布版 Needle 2 的内置权重 | 是，运行 `needle fetch` 获取 |
| 基础 checkpoint `checkpoints/*.pkl` | JAX/Flax 参考模型的参数，用于训练、评估或导出；不是原生运行时直接读取的文件 | 否，微调时再下载/准备 |
| SentencePiece tokenizer `tokenizer.model` / `tokenizer.vocab` | 把文本转换为训练和参考解码使用的 token；导出时会嵌入 `.cact` | 否，训练/参考解码时按需获取 |
| `.cact` | 将 checkpoint（可含 LoRA 合并结果）量化并打包为原生引擎可加载的调优权重 | 否；使用调优模型时才需要 |

因此，CPU 首次推理只需原生引擎。`needle fetch` 只负责当前机器的引擎，
不会偷偷替你下载训练 checkpoint；缺少训练资产时，请按相应教程显式准备。

## 显式获取原生引擎

激活 `.venv` 后，在联网机器上执行：

```sh
needle fetch
```

命令会根据操作系统和 CPU 架构自动选择构建，写入默认缓存
`~/.cache/cactus-needle/2.0.3/`，并打印类似以下信息（路径和扩展名随平台变化）：

```text
  engine    /home/alice/.cache/cactus-needle/2.0.3/libneedle.so
  deploy    copy to ~/.cache/cactus-needle/2.0.3/ on the device, or point NEEDLE_LIB_PATH at the file
```

检查文件确实存在且大小合理：

```sh
CACHE_DIR="$HOME/.cache/cactus-needle/2.0.3"
find "$CACHE_DIR" -maxdepth 1 -type f -printf '%f %s bytes\\n' 2>/dev/null || \
  find "$CACHE_DIR" -maxdepth 1 -type f -print
```

需要为另一台设备预取时，可显式指定 wheel tag，例如：

```sh
needle fetch --platform-tag manylinux2014_aarch64 --out ./engine-cache
```

这是跨设备部署的高级用法。若要获取独立的 engine runner（而不是 Python
包使用的共享库），使用 `needle download <platform>`，例如
`needle download linux-x86_64 --out ./runner`；可用的平台列表以
`needle --help` 和命令报错提示为准。

若要清除某个版本的引擎缓存，请先确认目录只包含 Needle 文件，再删除这个
明确的版本目录，之后重新执行 `needle fetch`：

```sh
rm -rf "$HOME/.cache/cactus-needle/2.0.3"
```

不要删除整个 `~/.cache`，也不要把 checkpoint 或 `.cact` 混放进引擎缓存目录。

## 在线后离线检查

先在线完成 `needle fetch`，再在同一环境中打开 Hugging Face 离线开关。这样
可以确认运行时只使用缓存，不会在资产缺失时隐式发起网络请求：

```sh
needle fetch
HF_HUB_OFFLINE=1 python -c "import needle; print('import ok:', needle.__version__)"
HF_HUB_OFFLINE=1 python - <<'PY'
import needle

agent = needle.Needle(tools=[])
result = agent.complete("hello", max_new_tokens=16)
assert isinstance(result, dict)
print("offline inference envelope:", result.get("type"), result.get("function_calls"))
PY
```

成功标准是命令退出码为 0，最后一行打印一个响应 envelope（即使没有工具，
`function_calls` 也应是空列表）。不要比较精确文本，因为模型版本、平台和
采样设置可能影响输出。若缓存缺失，关闭离线变量后重新执行 `needle fetch`；
不要依赖 `Needle` 初始化时的隐式下载来修复环境。设置了 `HF_HUB_OFFLINE=1`
时，缺失引擎应快速报错并指出 Hugging Face 离线限制。

`NEEDLE_LIB_PATH=/path/to/libneedle.so` 可以覆盖默认查找路径，适合把缓存文件
部署到自定义目录；使用它时仍建议先用上面的 `find` 检查文件存在。

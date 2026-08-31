# 首次推理

本页给出一条 CPU-first 的首次推理路径。先完成[安装与资产准备](installation.md)，
再选择 CLI 参考模型命令或原生 `Needle` Python API。两条路径都使用固定短提示和
有限输出长度；验收只看退出码、响应非空，不比较某一段精确文本。

## 前置条件

在仓库根目录激活 `.venv`，并先显式获取当前平台的原生引擎：

```sh
source .venv/bin/activate                 # Windows 请使用 installation.md 中的激活命令
needle fetch
```

`needle fetch` 下载的是原生引擎。`needle run` 还需要 JAX/Flax 参考模型的
checkpoint；如果本地没有指定文件，下面的 `load-checkpoint.pkl` 会由当前
`needle/model/run.py` 按 Hugging Face 路径尝试下载。生产应用通常直接使用下面的
`Needle` API 和已获取的原生引擎。

## CLI 参考模型

CLI 的 `run` 子命令用于加载 `.pkl` checkpoint，并在 CPU 上运行参考模型：

```sh
needle run \
  --checkpoint checkpoints/load-checkpoint.pkl \
  --query "用一句话介绍 Needle。" \
  --max-len 16 \
  --temperature 0
```

成功时会先打印 `prompt: ...`，随后打印一段可能为空格或标点开头的生成文本，
进程退出码为 `0`。`--max-len 16` 是 CLI 的生成上限（对应参考实现的
`max_new_tokens` 概念），`--temperature 0` 固定为 greedy 解码。模型版本、
checkpoint 和平台不同，文本内容可以不同；请检查命令成功退出且终端出现输出，
不要把示例文本作为快照断言。

如果看到 `FileNotFoundError`、checkpoint 格式错误或下载失败：确认网络可用，
并检查路径确实是 Needle format-v2 的 `.pkl`。`needle fetch` 只修复原生引擎
缺失，不会替代 checkpoint 下载。

## Needle Python API（原生引擎）

CLI 验证后，用同一个短提示调用生产侧 API：

```sh
HF_HUB_OFFLINE=1 python - <<'PY'
import needle

agent = needle.Needle()
response = agent.complete("用一句话介绍 Needle。", max_new_tokens=16)
assert isinstance(response, dict)
assert response.get("type") in {"call", "text", "respond", "refuse"}
print("response type:", response.get("type"))
print("function calls:", response.get("function_calls") or [])
PY
```

这里的 `max_new_tokens=16` 会限制原生引擎本次请求的输出长度。没有工具时，
响应通常包含 `type` 和空的 `function_calls`；不同引擎版本可能返回 `text`、
`respond` 或 `refuse`，因此只检查 envelope 是字典且类型属于公开响应类型。

若引擎或缓存资产缺失，初始化或 `complete()` 会抛出 `RuntimeError`，错误通常会
指向 Hugging Face 下载或共享库加载。退出离线模式后重新运行 `needle fetch`，
再重试上面的检查；不要在离线模式下期待隐式下载。

完整的工具调用、`run()` 循环和结构化提取 API 见 [API 文档](apis.md)。

## 离线验收

在线执行过 `needle fetch` 后，可以用下面的命令确认不需要网络：

```sh
HF_HUB_OFFLINE=1 python -c "import needle; print('import ok:', needle.__version__)"
```

命令退出码为 `0` 即表示 Python 包和原生引擎缓存可被找到；要验证真正的请求，
再运行上面的 Python API 示例。若失败，清除错误的自定义 `NEEDLE_LIB_PATH`，
或关闭 `HF_HUB_OFFLINE` 后重新执行 `needle fetch`。

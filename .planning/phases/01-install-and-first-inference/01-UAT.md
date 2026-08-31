---
status: complete
phase: 01-install-and-first-inference
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md]
started: 2026-08-31T14:40:00Z
updated: 2026-08-31T14:48:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Chinese-first CPU installation guide with uv environment creation and train/test extras
expected: 按文档创建 uv 环境并安装 `.[train,test]`，依赖检查通过。
result: pass
source: automated
coverage_id: D1

### 2. Explicit engine fetch, cache inspection, and HF_HUB_OFFLINE inference check
expected: `needle fetch` 获取原生引擎，随后离线导入和请求检查通过。
result: pass
source: automated
coverage_id: D2

### 3. README Chinese installation entry links to doc/installation.md and preserves API quickstart
expected: README 安装入口链接有效且保留现有 API 上手内容。
result: pass
source: automated
coverage_id: D3

### 4. CLI reference inference with fixed prompt and bounded max-len
expected: `needle run` 使用固定短提示和长度上限运行成功并产生非空输出。
result: pass
source: automated
coverage_id: I1

### 5. Native API CPU response after explicit fetch and offline mode
expected: `Needle.complete(..., max_new_tokens=16)` 返回有效响应 envelope 并以退出码 0 结束。
result: pass
source: automated
coverage_id: I2

### 6. Documentation drift smoke checks
expected: 文档命令、README 链接、CLI/API 和 typed tool 入口均由 pytest 检查通过。
result: pass
source: automated
coverage_id: I3

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

None.

---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-08-31T14:03:01.105Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | todo | doc/installation.md | 5 | CUDA and Metal installation and verification are intentionally deferred to a later phase. | open |  | 2026-08-31T14:03:01.105Z |  |

````json
[
  {
    "id": 1,
    "kind": "todo",
    "phase": "01",
    "file": "doc/installation.md",
    "line": 5,
    "description": "CUDA and Metal installation and verification are intentionally deferred to a later phase.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-31T14:03:01.105Z",
    "resolved_at": null
  }
]
````

# Environments

Pre-configured tool surfaces in `needle.environments`. Each module is a hand-curated set of at most five tools whose enums, bounds, and descriptions map cleanly onto Needle's constrained decoding, plus a ready agent and a frozen acceptance suite.

```python
from needle.environments import smart_home

smart_home.agent.complete("dim the study lights to 30 percent")
smart_home.run_tests()
```

```sh
python -m needle.environments.smart_home    # run one frozen suite, exit 0 on pass
```

## Module contract

Every environment module exposes the same surface:

| Attribute | What it is |
| --- | --- |
| `TOOLS` | The decorated tool functions, each returning `{"ok": True, ...}` echoes of its arguments. |
| `SYSTEM` | The system prompt the agent is built with. |
| `agent` | A `needle.Needle(tools=TOOLS, system=SYSTEM)`, constructed lazily on first access so importing never fetches the engine. Construction sets `NEEDLE_STRICT_VALIDATE=1` (if unset) so out-of-bounds calls are suppressed to refusals. |
| `TEST_CASES` | The frozen suite: dicts of `query`, expected `calls`, `category`, and optional `critical`. |
| `run_tests(min_confidence=0.0, verbose=True)` | Runs the suite against the shipped engine; returns `True` at >=90% pass with zero critical failures. |

`needle.environments.ENVIRONMENTS` maps name to module; `needle.environments.run_tests()` runs all six.

## Available environments

### smart_home

Lights, thermostat, fans, blinds, robot vacuum, over four rooms (`kitchen`, `living_room`, `bedroom`, `study`).

| Tool | Arguments |
| --- | --- |
| `control_lights` | `room`, `action` on/off/dim, `brightness_percent` 0-100, `color` from 5 names |
| `set_thermostat` | `temperature` 10-30 C |
| `control_fan` | `room` (no kitchen), `action` on/off, `speed` low/medium/high |
| `control_blinds` | `room`, `action` open/close |
| `start_robot_vacuum` | `action` start/stop/dock, `room` (no study) |

### media_player

Smart speaker playback and volume. `play_music` starts named content; the zero-argument playback tools control whatever is already playing.

| Tool | Arguments |
| --- | --- |
| `play_music` | `query` verbatim, 1-80 chars |
| `pause_media`, `resume_media`, `skip_track` | none |
| `set_volume` | `level` 0-100 |

### productivity

Timers, reminders, calendar events, tasks, notes. Date and time phrases are copied verbatim for the host app to resolve; a reminder without a time is treated as incomplete, and undated to-dos belong to `add_task`.

| Tool | Arguments |
| --- | --- |
| `set_timer` | `time_human` verbatim, 1-60 chars |
| `create_reminder` | `message` 1-120 chars, `date_time_human` |
| `create_calendar_event` | `title` 1-120 chars, `start_time_human`, `location` 1-80 chars |
| `add_task` | `title` 1-120 chars, `priority` low/medium/high |
| `create_note` | `text` 1-200 chars, `title` 1-60 chars |

### wearable

Watch notifications, workouts, find-my-phone. Replies only go to the four named senders (`Maya`, `Leo`, `Dr. Patel`, `Cactus Team`); workout types use gerund forms.

| Tool | Arguments |
| --- | --- |
| `reply_to_notification` | `notification_match` sender enum, `text` verbatim, 1-240 chars |
| `dismiss_notification` | `notification_match` sender enum |
| `start_workout` | `workout_type` running/walking/cycling/swimming/strength/yoga |
| `end_workout`, `find_my_phone` | none |

### kitchen_appliance

Oven, coffee maker, dishwasher, cooking timers. Hard numeric bounds make unsafe requests unrepresentable; the one read-only tool keeps checks separate from actions.

| Tool | Arguments |
| --- | --- |
| `set_oven` | `temperature` 50-250 C |
| `control_coffee_maker` | `action` brew/stop/warm |
| `start_dishwasher` | `cycle` eco/heavy/quick |
| `set_cooking_timer` | `label` 1-40 chars, `minutes` 1-360 |
| `get_appliance_status` | `appliance` oven/coffee_maker/dishwasher, read-only |

### data_capture

Contacts, expenses, meals, water, weight: dictated facts into typed records. Every value is copied verbatim, a member of a closed set, or a bounded number; the phone pattern and email format are enforced during decoding.

| Tool | Arguments |
| --- | --- |
| `create_contact` | `name` 1-60 chars, `phone` pattern `^\+?[0-9][0-9 -]{5,17}$`, `email` format email |
| `log_expense` | `amount` 0-100000, `category` from 6, `merchant` 1-60 chars |
| `log_meal` | `description` 1-120 chars, `meal_type` breakfast/lunch/dinner/snack |
| `log_water_intake` | `amount_ml` 1-5000 |
| `log_weight` | `weight_kg` 20-300 |

## The frozen suites

Each suite has 32 cases across six categories:

| Category | Expectation |
| --- | --- |
| `positive` | The exact call, arguments included. |
| `missing` | `[]`: a required value was not stated, do not guess it. |
| `irrelevant` | `[]`: no declared tool covers the request. |
| `negation` | `[]`: the request is negated ("don't", "do not", "never"). |
| `invalid` | `[]`: a stated value is outside the declared bounds. |
| `parallel` | Two calls from one query, order-insensitive. |

Cases marked `critical: True` (all `missing`, `negation`, and `invalid` cases) fail the suite regardless of the overall rate. `run_tests()` scores raw model output; `run_tests(min_confidence=0.4)` applies the production contract, acting on a call only at or above the threshold and treating anything below as a refusal.

A suite scores the model's output, not the harness: each case is decided by what the model emitted for that query, so a red suite reports the model's fit to the tool surface rather than a defect in the code that scores it. The rule above is pinned by a test in `tests/test_environments.py` that runs without the engine, so it cannot change silently.

The shipped base model does not pass all six suites, measured on engine 2.0.4: five fall short, and the failures concentrate in the `missing`, `negation`, and `invalid` categories, where the model supplies a value the query never stated, drops a stated bound, or substitutes a value that is not in the enum. Whether a green suite is the intended target for the shipped base model is a question about the model rather than about this package; this page records the measurement instead of settling it.

## Adapting one

Swap the `Literal` values (rooms, contacts, categories) for your own and keep the shapes: closed sets as enums, bounded numbers, verbatim copy for free text, five tools or fewer. One learned rule from smart_home: avoid enum values that hide inside likely query words (a room named office poisons an off action, so that home has a study).

---
dataset_info:
  features:
  - name: query
    dtype: string
  - name: answers
    dtype: string
  - name: tools
    dtype: string
  - name: source
    dtype: string
  - name: model
    dtype: string
  splits:
  - name: train
    num_examples: 2996257
  - name: validation
    num_examples: 10000
  download_size: 2160782760
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- tool-calling
- function-calling
- on-device
- assistant
- synthetic
pretty_name: On-Device Tool Calls
size_categories:
- 1M<n<10M
---

# On-Device Tool Calls

A large-scale synthetic dataset for training compact tool-calling models. Designed for on-device assistants that run on phones, wearables, smart home hubs, and laptops — where latency and model size matter.

Built for [Needle](https://github.com/Cactus-Compute/needle), a 32M-parameter encoder-decoder transformer that achieves 92% single-call F1 and 78% multi-call F1 on tool-call generation.

## Overview

| | |
|---|---|
| **Examples** | 2,996,257 train / 10,000 validation |
| **Unique tool names** | ~1.75M (232 curated + millions from open-source sources) |
| **Tool categories** | 23 on-device domains |
| **Languages** | English (primary), some multilingual queries |
| **License** | Apache 2.0 |

## Schema

Each example has 5 fields:

| Field | Type | Description |
|---|---|---|
| `query` | string | Natural language user request (e.g. *"Set a timer for 10 minutes"*) |
| `tools` | string | JSON array of available tool definitions with names, descriptions, and parameter schemas |
| `answers` | string | JSON array of tool calls the assistant should make (e.g. `[{"name": "set_timer", "arguments": {"time_human": "10 minutes"}}]`) |
| `source` | string | Data source identifier |
| `model` | string | Model used for generation |

### Example

```json
{
  "query": "Turn off the bedroom lights and lock the front door",
  "tools": "[{\"name\":\"control_lights\",\"description\":\"Turn lights on or off in a specific room.\",\"parameters\":{\"room\":{\"type\":\"string\"},\"action\":{\"type\":\"string\"}}},{\"name\":\"lock_door\",\"description\":\"Lock or unlock a smart door lock.\",\"parameters\":{\"door\":{\"type\":\"string\"},\"action\":{\"type\":\"string\"}}}]",
  "answers": "[{\"name\":\"control_lights\",\"arguments\":{\"room\":\"bedroom\",\"action\":\"off\"}},{\"name\":\"lock_door\",\"arguments\":{\"door\":\"front door\",\"action\":\"lock\"}}]",
  "source": "synth-gemini-assistant",
  "model": "gemini-3.1-flash-lite-preview"
}
```

## Data Sources

The dataset is a blend of four sources, each contributing different strengths:

### synth-gemini-assistant (1.10M examples, 36.8%)

Custom synthetic data generated with Gemini Flash Lite using a curated pool of **232 tools across 23 on-device categories**. Each batch is generated with a randomized subset of tools and a specified call type (single, multi, or no-call). Queries span ultra-terse commands (*"timer 5"*), conversational requests, indirect intent (*"it's dark in here"*), and multilingual input.

**Tool categories (232 tools):**

| Category | Tools | Examples |
|---|---|---|
| Time & Productivity | 15 | `set_timer`, `set_alarm`, `create_reminder`, `create_calendar_event` |
| Lists & Notes | 21 | `create_list_item`, `create_note`, `mark_list_item_done`, `pin_note` |
| Messaging | 10 | `send_instant_message`, `make_phone_call`, `send_email`, `send_sms` |
| Device Control | 14 | `set_brightness`, `set_volume`, `toggle_wifi`, `toggle_bluetooth` |
| Media | 14 | `play_music`, `pause_media`, `play_podcast`, `cast_media` |
| Navigation | 9 | `get_directions`, `find_nearby`, `share_location`, `start_navigation` |
| Smart Home | 12 | `set_thermostat`, `control_lights`, `lock_door`, `start_robot_vacuum` |
| Utility | 15 | `evaluate_js`, `web_search`, `translate_text`, `get_weather` |
| Camera & Photos | 10 | `take_photo`, `record_video`, `share_photo`, `search_photos` |
| Fitness & Health | 10 | `start_workout`, `log_water_intake`, `get_step_count`, `log_meal` |
| System | 7 | `toggle_airplane_mode`, `clear_notifications`, `check_battery` |
| Finance | 10 | `send_payment`, `check_balance`, `convert_currency`, `get_stock_price` |
| Reading & News | 10 | `get_news_headlines`, `read_aloud`, `summarize_page`, `open_ebook` |
| Accessibility | 4 | `set_font_size`, `toggle_screen_reader`, `toggle_magnifier` |
| Shopping | 3 | `search_product`, `add_to_cart`, `check_order_status` |
| Social | 10 | `post_status_update`, `like_post`, `comment_on_post`, `follow_user` |
| Ride & Delivery | 6 | `request_ride`, `order_food`, `track_food_delivery` |
| File Management | 7 | `open_file`, `share_file`, `move_file`, `compress_files` |
| Wearable | 10 | `check_heart_rate`, `find_my_phone`, `change_watch_face` |
| Desktop | 13 | `split_screen`, `toggle_vpn`, `print_document`, `kill_process` |
| Browser | 6 | `open_tab`, `bookmark_page`, `clear_browsing_data` |
| Emergency & Safety | 8 | `call_emergency_services`, `share_emergency_location` |
| Digital Wellbeing | 8 | `check_screen_time`, `set_app_timer`, `toggle_bedtime_mode` |

### synth-pleias (1.00M examples, 33.5%)

Reprompted from [Pleias](https://huggingface.co/Pleias) open-source tool-calling data. Original examples were reprompted through Gemini to normalize tool schemas and query styles to match the on-device assistant domain. Provides diverse query patterns and broader tool coverage.

### synth-gcs (832K examples, 27.8%)

Reprompted from [Toucan](https://huggingface.co/datasets/Salesforce/Toucan) (Salesforce) tool-calling data via Gemini. Contributes high multi-call density (68.8% multi-call) and diverse tool combinations from the Salesforce ecosystem.

### xlam-function-calling-60k (58K examples, 1.9%)

Human-curated examples from [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (Salesforce). High-quality ground truth with verified tool calls. Provides a quality anchor for the synthetic majority.

## Distribution

### Call Type

```
Single-call (1 tool):  1,064,063  (35.5%)
Multi-call (2+ tools): 1,761,740  (58.8%)
No-call (empty []):      170,454  ( 5.7%)
```

### Calls Per Example

```
0 calls:    170,454  ( 5.7%)   ░░░░░░
1 call:   1,064,063  (35.5%)   ████████████████████████████████████
2 calls:  1,445,299  (48.2%)   ████████████████████████████████████████████████
3 calls:    261,474  ( 8.7%)   █████████
4 calls:     38,057  ( 1.3%)   █
5+ calls:    16,910  ( 0.6%)   ░
```

### Available Tools Per Example

The tool count distribution has been rebalanced from its original skewed form (28% at 3 tools, 27% at 10 tools) to a near-uniform distribution across 1-10 tools. This was done by trimming uncalled tools from over-represented bins while preserving all called tools and answer correctness.

```
 0 tools:   101,974  ( 3.4%)   ███
 1 tool:    334,077  (11.1%)   ███████████
 2 tools:   361,916  (12.1%)   ████████████
 3 tools:   280,541  ( 9.4%)   █████████
 4 tools:   234,639  ( 7.8%)   ████████
 5 tools:   280,186  ( 9.4%)   █████████
 6 tools:   284,218  ( 9.5%)   █████████
 7 tools:   292,442  ( 9.8%)   ██████████
 8 tools:   281,952  ( 9.4%)   █████████
 9 tools:   278,085  ( 9.3%)   █████████
10 tools:   266,224  ( 8.9%)   █████████
```

### Call Type by Source

| Source | Single | Multi | No-call |
|---|---|---|---|
| synth-gemini-assistant | 31.2% | 60.0% | 8.8% |
| synth-pleias | 43.8% | 49.4% | 6.8% |
| synth-gcs | 30.6% | 68.8% | 0.6% |
| xlam | 46.0% | 54.0% | 0.0% |

### Field Lengths (characters)

| Field | Mean | Median | P95 |
|---|---|---|---|
| query | 210 | 121 | 670 |
| answers | 210 | 161 | 510 |
| tools | 1,936 | 1,884 | 3,611 |

## Intended Use

This dataset is designed for training **small, fast tool-calling models** (10M-200M parameters) that run on-device. The model receives a user query and a set of available tool definitions, then generates the appropriate JSON tool calls.

**Target architecture:** Encoder-decoder transformer where the encoder processes `[query <tools> tool_definitions]` and the decoder generates `[<tool_call> JSON_tool_calls]`.

**Not intended for:** General-purpose language modeling, chat, or instruction following.

## Known Limitations

- **Synthetic data:** ~98% of examples are LLM-generated. While filtered for schema validity, some argument values may be unrealistic.
- **English-centric:** Queries are predominantly English with occasional multilingual examples.
- **Multi-call with few tools:** The hardest setting (1-3 tools, 2+ calls with different argument values) is underrepresented relative to its difficulty. Targeted generation of these examples is ongoing.
- **No audio:** The dataset is text-only. Audio-to-tool-call requires separate speech data.

## Citation

```bibtex
@misc{cactus-compute-tool-calls-2025,
  title={On-Device Tool Calls: A Large-Scale Synthetic Dataset for Compact Tool-Calling Models},
  author={Cactus Compute},
  year={2025},
  url={https://huggingface.co/datasets/Cactus-Compute/tool-calls}
}
```

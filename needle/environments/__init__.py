"""Pre-configured environments: hand-curated tool surfaces whose enums,
bounds, and descriptions map cleanly onto Needle's constrained decoding.
Each module ships TOOLS, SYSTEM, a ready agent, and a frozen acceptance
suite runnable as `python -m needle.environments.<name>`.

Note: The frozen acceptance suites are aspirational targets designed for
fine-tuning. They are not expected to pass out-of-the-box on the base
shipped engine without fine-tuning on the environment's tools.

    from needle.environments import smart_home

    smart_home.agent.complete("dim the study lights to 30 percent")
    smart_home.run_tests()

To adapt an environment to your product, swap the Literal values (rooms,
contacts, categories) for your own and keep the shapes: closed sets as
enums, bounded numbers, verbatim copy for free text, five tools or fewer.
"""

import importlib

_NAMES = ("data_capture", "kitchen_appliance", "media_player", "productivity",
          "smart_home", "wearable")

__all__ = ["ENVIRONMENTS", "run_tests", *_NAMES]


def _load(name):
    return importlib.import_module(f"needle.environments.{name}")


def __getattr__(name):
    if name in _NAMES:
        return _load(name)
    if name == "ENVIRONMENTS":
        return {n: _load(n) for n in _NAMES}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), "ENVIRONMENTS", *_NAMES])


def run_tests(min_confidence=0.0, verbose=True):
    passed = True
    for name in _NAMES:
        print(name)
        passed = _load(name).run_tests(min_confidence, verbose) and passed
    return passed

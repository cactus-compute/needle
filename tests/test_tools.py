import enum
import sys
import typing

import pytest

from needle import tool, Field
from needle.agent.tools import build_schema, pydantic_schema, _is_pydantic_model

pep604 = pytest.mark.skipif(sys.version_info < (3, 10),
                            reason="PEP 604 unions need Python 3.10+")


def test_basic_schema_types_and_required():
    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    schema = add._needle_tool
    assert schema["name"] == "add"
    assert schema["description"] == "Add two numbers."
    assert schema["parameters"]["properties"]["a"] == {"type": "integer"}
    assert set(schema["parameters"]["required"]) == {"a", "b"}


def test_defaults_are_optional():
    def f(city: str, units: str = "metric"):
        pass

    schema = build_schema(f)
    assert schema["parameters"]["required"] == ["city"]
    assert "units" in schema["parameters"]["properties"]


def test_literal_list_and_dict_types():
    def f(mode: typing.Literal["fast", "slow"], items: list, meta: dict):
        pass

    props = build_schema(f)["parameters"]["properties"]
    assert props["mode"] == {"type": "string", "enum": ["fast", "slow"]}
    assert props["items"]["type"] == "array"
    assert props["meta"] == {"type": "object"}


def test_typed_list_items():
    def f(tags: typing.List[int]):
        pass

    props = build_schema(f)["parameters"]["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "integer"}}


def test_optional_annotation_not_required():
    def f(a: str, b: typing.Optional[int] = None):
        pass

    schema = build_schema(f)
    assert schema["parameters"]["required"] == ["a"]


@pep604
def test_pep604_none_union_not_required():
    def f(a: str, b: str | None = None, c: int | None = None, d: str | int | None = None):
        pass

    schema = build_schema(f)
    assert schema["parameters"]["required"] == ["a"]
    props = schema["parameters"]["properties"]
    assert props["b"] == {"type": "string"}
    assert props["c"] == {"type": "integer"}
    assert props["d"] == {"type": "string"}


@pep604
def test_pep604_union_matches_typing_union():
    def f(value: int | str):
        pass

    def g(value: typing.Union[int, str]):
        pass

    assert build_schema(f)["parameters"] == build_schema(g)["parameters"]
    assert build_schema(f)["parameters"]["properties"]["value"] == {"type": "integer"}


def test_field_constraints_and_docstring_args():
    def f(temp: int = Field(description="temperature", ge=0, le=100),
          name: str = Field(default="x", pattern="^[a-z]+$", min_length=1)):
        """Set the thing.

        Args:
            temp: the temperature in C
        """

    schema = build_schema(f)
    props = schema["parameters"]["properties"]
    assert props["temp"]["minimum"] == 0 and props["temp"]["maximum"] == 100
    assert props["temp"]["description"] == "temperature"
    assert props["name"]["pattern"] == "^[a-z]+$"
    assert props["name"]["minLength"] == 1
    assert "temp" in schema["parameters"]["required"]
    assert "name" not in schema["parameters"].get("required", [])


def test_docstring_description_falls_back_to_args():
    def f(city: str):
        """Look up weather.

        Args:
            city: the city to look up
        """

    props = build_schema(f)["parameters"]["properties"]
    assert props["city"]["description"] == "the city to look up"


def test_pydantic_model_schema():
    import pydantic

    class Weather(pydantic.BaseModel):
        "Weather query."
        city: str
        units: str = "metric"

    assert _is_pydantic_model(Weather)
    schema = pydantic_schema(Weather)
    assert schema["name"] == "Weather"
    assert "city" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["city"]
    assert schema["description"] == "Weather query."


def test_tool_decorator_preserves_callable():
    @tool
    def greet(name: str) -> str:
        "Greet someone."
        return "hi " + name

    assert greet("bob") == "hi bob"
    assert greet._needle_tool["name"] == "greet"


def test_numeric_literal_and_int_enum_schema():
    class Level(enum.IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    def f(
        port: typing.Literal[80, 443, 8080],
        level: Level,
        mode: typing.Literal["fast", "slow"] = "fast"
    ):
        pass

    props = build_schema(f)["parameters"]["properties"]
    assert props["port"] == {"type": "integer", "enum": [80, 443, 8080]}
    assert props["level"] == {"type": "integer", "enum": [1, 2, 3]}
    assert props["mode"] == {"type": "string", "enum": ["fast", "slow"]}

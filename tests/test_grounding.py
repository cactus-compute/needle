import datetime
import json
from typing import List

import pydantic
import pytest


class _Stub:
    def __init__(self):
        self.envelopes = []
        self.calls = []

    def needle_init(self, system, tools, index):
        self.calls.append("init")
        return 0

    def needle_load(self, blob, size):
        return 0

    def needle_complete(self, text, *args):
        self.calls.append(("complete", text.decode("utf-8")))
        buffer = args[-2]
        envelope = self.envelopes.pop(0) if len(self.envelopes) > 1 else self.envelopes[0]
        buffer.value = json.dumps(envelope).encode("utf-8")
        return 0

    def needle_reset(self):
        self.calls.append("reset")


@pytest.fixture
def stub(monkeypatch, tmp_path):
    import needle

    engine = _Stub()
    base = tmp_path / "needle3.cact"
    base.write_bytes((0x05E12A84).to_bytes(4, "little") + b"base weights")
    monkeypatch.setattr(needle, "_lib", lambda generation=3: engine)
    monkeypatch.setattr(needle, "_library_path", lambda generation=3: "/tmp/libneedle3")
    monkeypatch.setattr(needle, "_base_weights_path", lambda generation: str(base))
    monkeypatch.setattr(needle, "_active", {})
    monkeypatch.setattr(needle, "_loaded_base", {})
    return engine


class Invoice(pydantic.BaseModel):
    vendor: str
    due_date: datetime.date
    total: float = 0.0


class LineItem(pydantic.BaseModel):
    price: float


class Order(pydantic.BaseModel):
    items: List[LineItem]


def set_thermostat(temperature: int, mode: str = "auto"):
    """Set the thermostat.

    Args:
        temperature: target temperature in Celsius
        mode: heating strategy to use
    """
    return {"temperature": temperature, "mode": mode}


def _invoice(total, due_date="2026-09-01"):
    return {"vendor": "Acme Corp", "total": total, "due_date": due_date}


def _envelope(arguments, name, ungrounded):
    return {"type": "call", "confidence": 0.9,
            "function_calls": [{"name": name, "arguments": arguments}],
            "validation": {"ungrounded": list(ungrounded), "negation": False}}


def _extract_envelope(arguments, ungrounded=("Invoice.total",), name="Invoice"):
    return _envelope(arguments, name, ungrounded)


def _run_envelope(arguments, ungrounded=("set_thermostat.temperature",)):
    return _envelope(arguments, "set_thermostat", ungrounded)


def _call(due_date, name="Invoice"):
    return {"type": "call", "confidence": 0.9,
            "function_calls": [{"name": name,
                                "arguments": {"vendor": "Acme", "due_date": due_date}}]}


def test_complete_flags_call_dates_that_contradict_the_input(stub):
    import needle

    stub.envelopes = [_call("2026-09-05")]
    agent = needle.Needle(tools=[Invoice])
    response = agent.complete("Send an invoice to Acme due on 5th September 2031")

    assert response["validation"]["ungrounded"] == ["Invoice.due_date"]


def test_complete_leaves_grounded_dates_alone(stub):
    import needle

    stub.envelopes = [_call("2031-09-05")]
    agent = needle.Needle(tools=[Invoice])
    response = agent.complete("Send an invoice to Acme due on 5th September 2031")

    assert "validation" not in response


def test_input_without_a_year_is_not_checked(stub):
    import needle

    stub.envelopes = [_call("2026-09-05")]
    agent = needle.Needle(tools=[Invoice])
    response = agent.complete("Send Acme an invoice due next Friday")

    assert "validation" not in response


def test_a_slash_date_does_not_mint_a_year_from_its_day_or_month(stub):
    import needle

    stub.envelopes = [_call("2024-05-06")]
    agent = needle.Needle(tools=[Invoice])
    response = agent.complete("Send an invoice to Acme due on 5/6/24")

    assert "validation" not in response
    assert needle._source_years("due on 5/6/24") == set()
    assert needle._source_years("due on 2024-05-06") == {2024}


def test_extract_accepts_a_date_written_with_slashes(stub):
    import needle

    stub.envelopes = [_call("2024-12-31")]
    invoice = needle.extract("Invoice from Acme Corp due 12/31/24", Invoice)

    assert invoice.due_date == datetime.date(2024, 12, 31)


def test_extract_licenses_a_relative_date_against_its_date_fact(stub):
    import needle

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    text = "Invoice from Acme for the March 2019 order, due tomorrow"
    stub.envelopes = [_call(tomorrow.isoformat())]
    assert "validation" not in needle.Needle(tools=[Invoice]).complete(text)

    invoice = needle.extract(text, Invoice)

    assert invoice.due_date == tomorrow


def test_run_refuses_ungrounded_calls_unless_strict_is_off(stub):
    import needle

    stub.envelopes = [_call("2026-09-05"), {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[Invoice])
    response = agent.run("Send an invoice to Acme due on 5th September 2031")
    assert response["results"] == [{"error": "ungrounded due_date"}]

    stub.envelopes = [_call("2026-09-05"), {"type": "respond", "function_calls": []}]
    agent.reset()
    response = agent.run("Send an invoice to Acme due on 5th September 2031", strict=False)
    assert response["results"][0].due_date == datetime.date(2026, 9, 5)


def test_system_facts_license_relative_dates(stub):
    import needle

    stub.envelopes = [_call("2026-07-22")]
    agent = needle.Needle(tools=[Invoice], system="date: 2026-07-21 Tue 14:30")
    response = agent.complete("invoice Acme tomorrow for the 2019 reunion")

    assert "validation" not in response


def test_years_carry_across_turns_until_reset(stub):
    import needle

    stub.envelopes = [_call("2031-09-05")]
    agent = needle.Needle(tools=[Invoice])
    assert "validation" not in agent.complete("bill Acme on 5th September 2031")
    assert "validation" not in agent.complete(json.dumps({"id": 42, "since": "2019-03-01"}))

    agent.reset()
    response = agent.complete("customer since March 2019, bill them")
    assert response["validation"]["ungrounded"] == ["Invoice.due_date"]


def test_engine_reported_fabrications_are_kept_and_block_execution(stub):
    import needle

    envelope = _call("2031-09-05")
    envelope["validation"] = {"ungrounded": ["Invoice.vendor"]}
    stub.envelopes = [envelope, {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[Invoice])
    response = agent.run("bill someone on 5th September 2031")

    assert response["results"] == [{"error": "ungrounded vendor"}]
    assert stub.calls[-1] == ("complete", json.dumps([{"error": "ungrounded vendor"}]))


def test_extract_clears_a_thousands_separated_number(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(1200.0))]
    invoice = needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)

    assert invoice.total == 1200.0
    assert invoice.due_date == datetime.date(2026, 9, 1)


def test_extract_clears_a_thousands_separated_integer(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(1200))]
    invoice = needle.extract("Invoice from Acme Corp, $1,200, due 2026-09-01", Invoice)

    assert invoice.total == 1200


def test_extract_leaves_an_unseparated_number_working(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(1200.0))]
    invoice = needle.extract("Invoice from Acme Corp, $1200.00, due 2026-09-01", Invoice)

    assert invoice.total == 1200.0


def test_extract_still_rejects_a_fabricated_number(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(9999.0))]
    with pytest.raises(needle.ExtractionValidationError) as excinfo:
        needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)

    assert "total" in str(excinfo.value)


def test_extract_still_rejects_a_non_numeric_engine_flag(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(1200.0),
                                        ungrounded=("Invoice.vendor",))]
    with pytest.raises(needle.ExtractionValidationError) as excinfo:
        needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)

    assert "vendor" in str(excinfo.value)


def test_extract_clears_a_nested_separated_number(stub):
    import needle

    stub.envelopes = [_extract_envelope({"items": [{"price": 1100.0}]},
                                        ungrounded=("Order.items[0].price",),
                                        name="Order")]
    order = needle.extract("Order with one item priced $1,100.00", Order)

    assert order.items[0].price == 1100.0


def test_extract_does_not_clear_a_digit_run_inside_a_longer_number(stub):
    import needle

    for source in ("Invoice from Acme Corp, $12,000.00, due 2026-09-01",
                   "Invoice from Acme Corp, $11,200.00, due 2026-09-01",
                   "Invoice from Acme Corp, $1,2000.00, due 2026-09-01"):
        stub.envelopes = [_extract_envelope(_invoice(1200.0))]
        with pytest.raises(needle.ExtractionValidationError):
            needle.extract(source, Invoice)


def test_extract_does_not_clear_a_number_absent_from_the_source(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(1200.0))]
    with pytest.raises(needle.ExtractionValidationError):
        needle.extract("Invoice from Acme Corp, due 2026-09-01", Invoice)


def test_extract_does_not_read_a_date_hyphen_as_a_minus_sign(stub):
    import needle

    for value in (-9.0, -1.0):
        stub.envelopes = [_extract_envelope(_invoice(value))]
        with pytest.raises(needle.ExtractionValidationError):
            needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01",
                           Invoice)


def test_extract_does_not_read_a_range_hyphen_as_a_minus_sign(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(-1400.0))]
    with pytest.raises(needle.ExtractionValidationError):
        needle.extract("Invoice from Acme Corp, $1,200.00-1,400.00, due 2026-09-01",
                       Invoice)


def test_extract_clears_a_written_negative(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(-1200.0))]
    invoice = needle.extract("Invoice from Acme Corp, -1,200.00, due 2026-09-01",
                             Invoice)

    assert invoice.total == -1200.0


def test_extract_clears_a_zero_against_its_source(stub):
    import needle

    stub.envelopes = [_extract_envelope(_invoice(0.0))]
    invoice = needle.extract("Invoice from Acme Corp, $0.00, due 2026-09-01", Invoice)

    assert invoice.total == 0.0


def test_run_executes_a_grounded_number_and_refuses_an_ungrounded_sibling(stub):
    import needle

    envelope = {"type": "call", "confidence": 0.9,
                "function_calls": [
                    {"name": "set_thermostat", "arguments": {"temperature": 21}},
                    {"name": "set_thermostat", "arguments": {"temperature": 22}}],
                "validation": {"ungrounded": ["set_thermostat.temperature"],
                               "negation": False}}
    stub.envelopes = [envelope, {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[set_thermostat])
    response = agent.run("make it 21 and cool the room")

    assert response["results"] == [{"temperature": 21, "mode": "auto"},
                                   {"error": "ungrounded temperature"}]


def test_run_executes_a_separated_number_grounded_in_the_query(stub):
    import needle

    stub.envelopes = [_run_envelope({"temperature": 1200}),
                      {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[set_thermostat])
    response = agent.run("set it to 1,200")

    assert response["results"] == [{"temperature": 1200, "mode": "auto"}]


def test_run_refuses_a_number_absent_from_the_query(stub):
    import needle

    stub.envelopes = [_run_envelope({"temperature": 99}),
                      {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[set_thermostat])
    response = agent.run("make it 21 and cool the room")

    assert response["results"] == [{"error": "ungrounded temperature"}]


def test_run_still_refuses_a_non_numeric_engine_flag(stub):
    import needle

    stub.envelopes = [_run_envelope({"temperature": 21},
                                    ungrounded=("set_thermostat.mode",)),
                      {"type": "respond", "function_calls": []}]
    agent = needle.Needle(tools=[set_thermostat])
    response = agent.run("make it 21 and cool the room")

    assert response["results"] == [{"error": "ungrounded mode"}]

from acb.agents.domain_agents import cards_agent, compliance_agent, loans_agent, mortgage_agent, payments_agent


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, **kwargs):
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text):
        self.models = _FakeModels(text)


class _ExplodingClient:
    class models:
        @staticmethod
        def generate_content(**kwargs):
            raise AssertionError("should not call the model")


def test_cards_agent_escalates_on_sentinel():
    cmd = cards_agent({"message": "my card was stolen"}, _FakeClient("ESCALATE"))
    assert cmd.goto == "escalation"
    assert "escalation_reason" in cmd.update


def test_payments_agent_finishes_with_reply():
    cmd = payments_agent({"message": "how do transfers work?", "jurisdiction": None}, _FakeClient("Here's how transfers work."))
    assert cmd.goto == "output_guardrail"
    assert cmd.update["agent_reply"] == "Here's how transfers work."
    assert cmd.update["agent_name"] == "payments"


def test_loans_agent_defers_to_compliance():
    cmd = loans_agent({"message": "what's the current loan rate?", "handoff_count": 0}, _FakeClient("DEFER_TO_COMPLIANCE"))
    assert cmd.goto == "supervisor"
    assert cmd.update["route_suggestion"] == "compliance"
    assert cmd.update["handoff_count"] == 1


def test_loans_agent_defer_increments_existing_handoff_count():
    cmd = loans_agent({"message": "what's the current loan rate?", "handoff_count": 2}, _FakeClient("DEFER_TO_COMPLIANCE"))
    assert cmd.update["handoff_count"] == 3


def test_mortgage_agent_asks_for_jurisdiction_before_quoting_without_calling_model():
    cmd = mortgage_agent(
        {"message": "what would a $300k mortgage at 6% over 30 years cost?", "language": "en"},
        _ExplodingClient(),
    )
    assert cmd.goto == "output_guardrail"
    assert "island" in cmd.update["agent_reply"].lower()


def test_mortgage_agent_proceeds_when_jurisdiction_known():
    cmd = mortgage_agent(
        {
            "message": "what would a $300k mortgage at 6% over 30 years cost?",
            "jurisdiction": "Antigua & Barbuda",
            "language": "en",
        },
        _FakeClient("Estimated payment based on the figures you gave..."),
    )
    assert cmd.goto == "output_guardrail"
    assert cmd.update["agent_name"] == "mortgage"


def test_compliance_agent_is_terminal():
    cmd = compliance_agent({"message": "what's the wire transfer fee?", "jurisdiction": None}, _FakeClient("Per the Antigua fee schedule, the wire fee is..."))
    assert cmd.goto == "output_guardrail"
    assert cmd.update["agent_name"] == "compliance"


def test_compliance_agent_can_still_escalate():
    cmd = compliance_agent({"message": "anything", "jurisdiction": None}, _FakeClient("ESCALATE"))
    assert cmd.goto == "escalation"

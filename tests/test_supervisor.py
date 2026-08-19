import json

from acb.agents.supervisor import MAX_HANDOFFS, SPECIALIST_ROUTES, route_after_guardrail, supervisor_node


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


def _decision(next_route, reasoning="because"):
    return json.dumps({"next": next_route, "reasoning": reasoning})


def test_supervisor_routes_to_specialist():
    state = {"message": "I need to activate my new debit card", "handoff_count": 0}
    cmd = supervisor_node(state, _FakeClient(_decision("cards")))
    assert cmd.goto == "cards"
    assert cmd.update["route"] == "cards"


def test_supervisor_falls_back_to_faq_on_garbage_output():
    state = {"message": "anything", "handoff_count": 0}
    cmd = supervisor_node(state, _FakeClient("not valid json"))
    assert cmd.goto == "faq"


def test_supervisor_falls_back_to_faq_on_unknown_route():
    state = {"message": "anything", "handoff_count": 0}
    cmd = supervisor_node(state, _FakeClient(_decision("not_a_real_route")))
    assert cmd.goto == "faq"


def test_supervisor_never_finishes_before_any_specialist_responded():
    # No agent_reply in state yet -> FINISH is not even in the schema's
    # enum, but a misbehaving model could still emit it; must not be honored.
    state = {"message": "anything", "handoff_count": 0}
    cmd = supervisor_node(state, _FakeClient(_decision("FINISH")))
    assert cmd.goto != "output_guardrail"
    assert cmd.goto == "faq"  # safe fallback


def test_supervisor_finishes_after_a_specialist_has_responded():
    state = {"message": "thanks, that's all", "agent_reply": "already answered", "handoff_count": 1}
    cmd = supervisor_node(state, _FakeClient(_decision("FINISH")))
    assert cmd.goto == "output_guardrail"


def test_supervisor_forces_escalation_at_handoff_cap_without_calling_model():
    class _ExplodingModels:
        def generate_content(self, **kwargs):
            raise AssertionError("should not call the model once the handoff cap is hit")

    class _ExplodingClient:
        models = _ExplodingModels()

    state = {"message": "anything", "agent_reply": "x", "handoff_count": MAX_HANDOFFS}
    cmd = supervisor_node(state, _ExplodingClient())
    assert cmd.goto == "escalation"


def test_all_specialist_routes_are_reachable_targets():
    for name in SPECIALIST_ROUTES:
        state = {"message": "anything", "handoff_count": 0}
        cmd = supervisor_node(state, _FakeClient(_decision(name)))
        assert cmd.goto == name


def test_route_after_guardrail_blocked():
    assert route_after_guardrail({"blocked": True}) == "blocked"


def test_route_after_guardrail_ok():
    assert route_after_guardrail({"blocked": False}) == "supervisor"

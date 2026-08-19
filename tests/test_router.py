from acb.agents.router import AGENT_NAMES, route_after_guardrail, route_to_agent, router_node


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


def test_router_node_returns_valid_category():
    state = {"message": "I need to activate my new debit card"}
    result = router_node(state, _FakeClient("cards"))
    assert result["route"] == "cards"


def test_router_node_falls_back_to_faq_on_unknown_category():
    state = {"message": "anything"}
    result = router_node(state, _FakeClient("not_a_real_category"))
    assert result["route"] == "faq"


def test_router_node_falls_back_to_faq_on_empty_response():
    state = {"message": "anything"}
    result = router_node(state, _FakeClient(""))
    assert result["route"] == "faq"


def test_all_agent_names_are_reachable_targets():
    # Every declared category must correctly round-trip as a valid route.
    for name in AGENT_NAMES:
        state = {"message": "anything"}
        result = router_node(state, _FakeClient(name))
        assert result["route"] == name


def test_route_after_guardrail_blocked():
    assert route_after_guardrail({"blocked": True}) == "blocked"


def test_route_after_guardrail_ok():
    assert route_after_guardrail({"blocked": False}) == "router"


def test_route_to_agent_uses_state_route():
    assert route_to_agent({"route": "loans"}) == "loans"


def test_route_to_agent_defaults_to_faq_when_missing():
    assert route_to_agent({}) == "faq"

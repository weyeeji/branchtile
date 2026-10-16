"""Explicit internal-to-external ID translation for the pinned native queue API."""


def request_transport(internal_ids, request_states):
    """LLM.enqueue returns LLMEngine.add_request's INTERNAL req_id in this build."""
    assert len(internal_ids) == len(set(internal_ids))
    external_ids = []
    transport = {}
    for internal in internal_ids:
        state = request_states[internal]
        assert state.request_id == internal
        external = state.external_req_id
        assert external is not None and external not in transport
        external_ids.append(external)
        transport[external] = internal
    return external_ids, transport

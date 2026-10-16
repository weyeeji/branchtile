"""Admit an entire ready request group before advancing vLLM 0.31 scheduler.

The ordinary LLM.generate loop sends requests one by one to a live engine.
A synchronous pause/ADD/resume message sequence prevents model speed or host
submission timing from changing which requests are present at the first step.
"""


def generate_ready_group(llm, prompts, sampling_params):
    core = llm.llm_engine.engine_core
    core.call_utility('pause_scheduler', 'keep', False)
    try:
        request_ids = llm.enqueue(prompts, sampling_params, use_tqdm=False)
        assert len(request_ids) == len(prompts)
    finally:
        core.call_utility('resume_scheduler')
    return llm.wait_for_completion(use_tqdm=False)

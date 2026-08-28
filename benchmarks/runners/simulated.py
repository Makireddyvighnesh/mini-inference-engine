from __future__ import annotations

import time
from typing import Any

from ..core.harness import RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec


def simulated_runner(
    request: RequestSpec,
    recorder: RequestEventRecorder,
    *,
    prefill_ms: float = 0.10,
    token_ms: float = 0.05,
) -> RequestOutcome:
    """A deterministic CPU runner used to validate the Phase 0 harness.

    It deliberately models the request-event contract without pretending to
    measure a neural network. Phase 1 replaces this callback with the pinned
    Hugging Face model runner.
    """

    if prefill_ms < 0 or token_ms < 0:
        raise ValueError("simulated delays must be non-negative")
    recorder.record("prefill_start")
    if prefill_ms:
        time.sleep(prefill_ms / 1000.0)
    recorder.record("prefill_end")

    output_ids = tuple(
        (request.prompt_token_ids[-1] + index + 1) % 32_000
        for index in range(request.max_new_tokens)
    )
    for token_index, token_id in enumerate(output_ids):
        if token_ms:
            time.sleep(token_ms / 1000.0)
        recorder.mark_token_ready(token_index, token_id=token_id)
        recorder.mark_token_sent(token_index, token_id=token_id)
    recorder.record("completion")
    return RequestOutcome(
        status="completed",
        generated_token_ids=output_ids,
        metadata={"runner": "simulated_cpu", "model_execution": False},
    )


def make_simulated_runner(
    *,
    prefill_ms: float = 0.10,
    token_ms: float = 0.05,
) -> Any:
    """Return a configured runner callback for CLI and tests."""

    def run(request: RequestSpec, recorder: RequestEventRecorder) -> RequestOutcome:
        return simulated_runner(
            request,
            recorder,
            prefill_ms=prefill_ms,
            token_ms=token_ms,
        )

    return run

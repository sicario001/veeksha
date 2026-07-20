"""Regression: the DispatchTracker must advance even when a client swallows a
transport failure into an error RequestResult without firing the dispatch/
prefill callbacks (the audio clients do exactly this on connect failures).
Before the fix, one such failure deadlocked every subsequent request under
sequential launch."""

from __future__ import annotations

import asyncio
import threading
from queue import Queue
from types import SimpleNamespace

from veeksha.core.response import RequestResult
from veeksha.traffic.dispatch_tracker import DispatchTracker
from veeksha.workers.client_runner import ClientWorker


class _SilentFailureClient:
    """Returns an error result WITHOUT firing any callback and without raising
    — the behavior of the audio clients on connect/timeout failures."""

    async def send_request(
        self,
        request,
        session_id,
        session_total_requests,
        on_request_sent,
        on_request_dispatched,
    ):
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            channels={},
            success=False,
            error_code=503,
            error_msg="connect refused",
        )


def _request(rid: int, ticket: int):
    return SimpleNamespace(id=rid, dispatch_ticket=ticket)


def test_tracker_advances_when_client_swallows_failure():
    tracker = DispatchTracker(ordering="dispatch")
    scheduler = SimpleNamespace(
        dispatch_tracker=tracker,
        notify_request_sent=lambda _rid: None,
    )
    output: Queue = Queue()
    worker = ClientWorker(
        worker_id=0,
        client=_SilentFailureClient(),
        input_queue=Queue(),
        output_queue=output,
        stop_event=threading.Event(),
        traffic_scheduler=scheduler,
    )

    async def _run():
        # ticket 0 fails silently; ticket 1 must still get its turn (it blocks
        # in wait_for_turn until ticket 0's completion advances the counter).
        item0 = (_request(0, 0), 0, 1, 0.0, 0.0)
        item1 = (_request(1, 1), 1, 1, 0.0, 0.0)
        await asyncio.wait_for(worker._process_request(item0), timeout=5.0)
        await asyncio.wait_for(worker._process_request(item1), timeout=5.0)

    asyncio.run(_run())
    assert output.qsize() == 2
    first = output.get_nowait()
    assert first.success is False

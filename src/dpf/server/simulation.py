"""SimulationManager — async lifecycle wrapper around SimulationEngine.

Manages the create -> start -> pause -> resume -> stop lifecycle and
broadcasts StepResult data to WebSocket subscriber queues.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import Any

import numpy as np

from dpf.config import SimulationConfig
from dpf.core.bases import StepResult
from dpf.engine import SimulationEngine
from dpf.server.models import SimulationStatus

logger = logging.getLogger(__name__)


class SimulationManager:
    """Manages one simulation's lifecycle.

    The simulation runs in an asyncio task, calling ``engine.step()`` in a
    tight loop.  Between steps it yields to the event loop so WebSocket
    handlers and REST endpoints remain responsive.

    Attributes:
        sim_id: Unique identifier for this simulation.
        status: Current lifecycle state.
        engine: Underlying SimulationEngine (None until created).
        last_result: Most recent StepResult from step().
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        max_steps: int | None = None,
        sim_id: str | None = None,
    ) -> None:
        self.sim_id = sim_id or uuid.uuid4().hex[:12]
        self.config = config
        self.max_steps = max_steps
        self.status = SimulationStatus.idle
        self.engine: SimulationEngine | None = None
        self.last_result: StepResult | None = None
        self.error_message: str | None = None

        # Async plumbing
        self._task: asyncio.Task[None] | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_flag = False

        # Subscriber queues (one per WebSocket client)
        self._subscribers: list[asyncio.Queue[StepResult]] = []

    # ── Public lifecycle API ─────────────────────────────────────

    def create_engine(self) -> None:
        """Instantiate the SimulationEngine (heavyweight, imports physics)."""
        self.engine = SimulationEngine(self.config)
        logger.info("Created engine for sim %s", self.sim_id)

    def subscribe(self) -> asyncio.Queue[StepResult]:
        """Register a subscriber queue for step-result broadcasts.

        Returns:
            An asyncio.Queue that will receive each StepResult.
        """
        q: asyncio.Queue[StepResult] = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[StepResult]) -> None:
        """Remove a subscriber queue."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(q)

    async def start(self) -> None:
        """Begin the simulation run loop in a background task."""
        if self.engine is None:
            self.create_engine()
        if self.status in (SimulationStatus.running,):
            return

        self._stop_flag = False
        self._pause_event.set()
        self.status = SimulationStatus.running
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Started sim %s", self.sim_id)

    async def pause(self) -> None:
        """Pause the simulation loop (can be resumed)."""
        if self.status != SimulationStatus.running:
            return
        self._pause_event.clear()
        self.status = SimulationStatus.paused
        logger.info("Paused sim %s at step %d", self.sim_id, self._step_count())

    async def resume(self) -> None:
        """Resume a paused simulation."""
        if self.status != SimulationStatus.paused:
            return
        self._pause_event.set()
        self.status = SimulationStatus.running
        logger.info("Resumed sim %s", self.sim_id)

    async def stop(self) -> None:
        """Stop the simulation permanently."""
        self._stop_flag = True
        self._pause_event.set()  # Unblock if paused
        if self._task is not None:
            await self._task
            self._task = None
        if self.status != SimulationStatus.error:
            self.status = SimulationStatus.finished
        logger.info("Stopped sim %s", self.sim_id)

    def get_field_snapshot(self) -> dict[str, np.ndarray]:
        """Return current field arrays (thread-safe copy)."""
        if self.engine is None:
            return {}
        return self.engine.get_field_snapshot()

    def info(self) -> dict[str, Any]:
        """Build a SimulationInfo-compatible dict."""
        r = self.last_result
        return {
            "sim_id": self.sim_id,
            "status": self.status.value,
            "step": r.step if r else 0,
            "time": r.time if r else 0.0,
            "current": r.current if r else 0.0,
            "voltage": r.voltage if r else 0.0,
            "energy_conservation": r.energy_conservation if r else 1.0,
            "max_Te": r.max_Te if r else 0.0,
            "max_rho": r.max_rho if r else 0.0,
            "total_radiated_energy": r.total_radiated_energy if r else 0.0,
            "error_message": self.error_message,
        }

    # ── Internal ─────────────────────────────────────────────────

    def _step_count(self) -> int:
        if self.engine is not None:
            return self.engine.step_count
        return 0

    async def _run_loop(self) -> None:
        """Async loop that calls engine.step() and broadcasts results."""
        assert self.engine is not None
        try:
            while not self._stop_flag:
                # Honor pause
                await self._pause_event.wait()
                if self._stop_flag:
                    break

                # Run one step (CPU-bound, so yield after)
                result = await asyncio.to_thread(
                    self.engine.step, _max_steps=self.max_steps,
                )
                self.last_result = result

                # Broadcast to subscribers
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(result)
                    except asyncio.QueueFull:
                        # Slow consumer — drop oldest
                        with contextlib.suppress(asyncio.QueueEmpty):
                            q.get_nowait()
                        with contextlib.suppress(asyncio.QueueFull):
                            q.put_nowait(result)

                if result.finished:
                    self.status = SimulationStatus.finished
                    break

                # Yield to event loop
                await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("Simulation %s error: %s", self.sim_id, exc)
            self.status = SimulationStatus.error
            self.error_message = str(exc)

        # Send final sentinel to wake any blocking consumers
        if self.last_result is not None:
            sentinel = StepResult(
                time=self.last_result.time,
                step=self.last_result.step,
                finished=True,
            )
            for q in list(self._subscribers):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(sentinel)

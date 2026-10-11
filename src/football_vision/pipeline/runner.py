"""Pipeline runner: ordering, caching, logging and timing of stages."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_vision.artifacts import ArtifactStore
from football_vision.config import PipelineConfig, dump_config
from football_vision.errors import PipelineError
from football_vision.eval.profiling import Timer
from football_vision.pipeline.stages import STAGE_ORDER, STAGE_REGISTRY, RunContext
from football_vision.utils import LOGGER_NAME, set_seed


@dataclass(frozen=True)
class StageResult:
    """Outcome of one stage in a runner invocation.

    Attributes:
        name: Stage name.
        status: ``"ran"`` if the stage executed, ``"skipped"`` if its outputs
            already existed.
        seconds: Wall-clock duration in seconds (0 for skipped stages).
    """

    name: str
    status: str
    seconds: float


class PipelineRunner:
    """Executes pipeline stages in dependency order with artifact caching.

    Attributes:
        cfg: Configuration of the run.
        ctx: Context shared by all stages.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        services: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Creates a runner and the run directory.

        Args:
            cfg: Configuration of the run; artifacts go to ``cfg.run_dir``.
            services: Optional injected dependencies such as ``{"detector": obj}``.
            logger: Logger to use; defaults to the package logger.
        """
        self.cfg = cfg
        self.ctx = RunContext(
            cfg=cfg,
            store=ArtifactStore(cfg.run_dir),
            logger=logger or logging.getLogger(LOGGER_NAME),
            services=dict(services or {}),
        )

    def run(
        self, stages: Sequence[str] | None = None, force: Iterable[str] | bool = False
    ) -> list[StageResult]:
        """Runs the requested stages.

        Stages always execute in the canonical order (``STAGE_ORDER``), whatever order
        they are requested in. A stage is skipped when all its artifacts already exist,
        unless it is forced. The resolved configuration and the timings are saved in
        the run directory.

        Args:
            stages: Stage names to run; ``None`` uses ``cfg.stages``.
            force: ``True`` re-runs every requested stage; an iterable of names
                re-runs only those stages.

        Returns:
            One :class:`StageResult` per requested stage, in execution order.

        Raises:
            PipelineError: If a stage name is unknown or a stage's required
                artifacts are missing.
        """
        requested = list(self.cfg.stages if stages is None else stages)
        unknown = [name for name in requested if name not in STAGE_REGISTRY]
        if unknown:
            raise PipelineError(
                f"Unknown stage(s) {unknown}; choose from {list(STAGE_ORDER)}"
            )
        force_all = force is True
        force_names = set() if force_all or force is False else set(force)
        set_seed(self.cfg.seed)
        store = self.ctx.store
        dump_config(self.cfg, store.root / "config.yaml")
        results: list[StageResult] = []
        for name in STAGE_ORDER:
            if name not in requested:
                continue
            stage = STAGE_REGISTRY[name](self.ctx)
            missing = [
                artifact for artifact in stage.requires if not store.exists(artifact)
            ]
            if missing:
                raise PipelineError(
                    f"Stage '{name}' needs artifacts {missing}; "
                    "run the stages that produce them first."
                )
            if not (force_all or name in force_names) and stage.outputs_exist():
                self.ctx.logger.info("Skipping '%s' (outputs exist)", name)
                results.append(StageResult(name, "skipped", 0.0))
                continue
            self.ctx.logger.info("Running '%s'", name)
            with Timer() as timer:
                stage.run()
            results.append(StageResult(name, "ran", timer.elapsed))
        timings = {
            r.name: {"status": r.status, "seconds": round(r.seconds, 3)}
            for r in results
        }
        (store.root / "timings.json").write_text(
            json.dumps(timings, indent=2), encoding="utf-8"
        )
        return results

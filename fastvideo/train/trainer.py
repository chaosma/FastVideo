# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.tracking import build_tracker
from fastvideo.training import memory_probe


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _maybe_save_on_cpu_ctx():
    """Wrap the forward+backward in PyTorch's default activation offload
    when ``FASTVIDEO_SAVE_ON_CPU=1``.

    Composition with the activation tracer
    --------------------------------------
    ``saved_tensors_hooks`` is winner-takes-all: only the innermost
    registered (pack, unpack) pair is used. The activation tracer
    (``ActivationTrace.__enter__`` in single_train_step) enters its own
    ``saved_tensors_hooks`` *after* this wrapper, so when both are on the
    tracer is innermost and these hooks are dormant --- the tracer's
    pack/unpack does the offload itself (see ``ActivationTrace._pack_hook``).

    This wrapper still runs (harmlessly) when the tracer is on; it
    provides the actual offload only when the tracer is disabled."""
    if os.environ.get("FASTVIDEO_SAVE_ON_CPU",
                      "").strip().lower() not in _TRUE_VALUES:
        return contextlib.nullcontext()
    pin = os.environ.get("FASTVIDEO_SAVE_ON_CPU_PIN",
                         "1").strip().lower() in _TRUE_VALUES
    return torch.autograd.graph.save_on_cpu(pin_memory=pin)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


def _coerce_log_scalar(
    value: Any,
    *,
    where: str,
) -> float | torch.Tensor:
    """Coerce *value* to a loggable scalar.

    GPU tensors stay on device so we avoid a
    ``cudaDeviceSynchronize`` per accumulation step.
    The caller must materialize them to float when logging.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor at {where}, "
                             f"got shape={tuple(value.shape)}")
        return value.detach()
    if isinstance(value, float | int):
        return float(value)
    raise TypeError(f"Expected a scalar (float/int/Tensor) at "
                    f"{where}, got {type(value).__name__}")


@dataclass(slots=True)
class TrainLoopState:
    step: int
    accum_iter: int


class Trainer:

    def __init__(
        self,
        training_config: TrainingConfig,
        *,
        config: dict[str, Any] | None = None,
        callback_configs: dict[str, dict[str, Any]]
        | None = None,
    ) -> None:
        self.training_config = training_config
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.local_rank = self.world_group.local_rank
        self.tracker = build_tracker(
            training_config.tracker,
            training_config.checkpoint,
            config=config,
        )
        self.callbacks = CallbackDict(
            callback_configs or {},
            training_config,
        )

    def _iter_dataloader(self, dataloader: Any) -> Iterator[dict[str, Any]]:
        data_iter = iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def run(
        self,
        method: TrainingMethod,
        *,
        dataloader: Any,
        max_steps: int,
        start_step: int = 0,
        checkpoint_manager: Any | None = None,
    ) -> None:
        tc = self.training_config
        grad_accum = max(
            1,
            int(tc.loop.gradient_accumulation_steps or 1),
        )

        method.set_tracker(self.tracker)
        method.on_train_start()
        self.callbacks.on_train_start(
            method,
            iteration=start_step,
        )

        resume_from_checkpoint = (tc.checkpoint.resume_from_checkpoint or "")
        if checkpoint_manager is not None:
            if resume_from_checkpoint:
                method.seed_optimizer_state_for_resume()
            resumed_step = (checkpoint_manager.maybe_resume(resume_from_checkpoint=(resume_from_checkpoint)))
            if resumed_step is not None:
                start_step = int(resumed_step)
        self.callbacks.on_validation_begin(
            method,
            iteration=start_step,
        )
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader)

        # Restore the RNG snapshot LAST — after dcp.load,
        # after iter(dataloader), after everything that may
        # have advanced the RNG as a side-effect.
        if (checkpoint_manager is not None and resume_from_checkpoint):
            checkpoint_manager.load_rng_snapshot(resume_from_checkpoint, )
        probe = memory_probe.maybe_init_from_env(self.global_rank)
        if probe is not None:
            transformer = getattr(method.student, "transformer", None)
            optimizer = next(iter(method.get_optimizers(start_step)), None)
            if transformer is not None:
                probe.install(transformer, optimizer=optimizer)

        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        for step in progress:
            memory_probe.step_begin()
            memory_probe.snap("P0_idle")
            t0 = time.perf_counter()

            # Accumulate on GPU during grad-accum; materialise
            # to CPU once per step right before logging.
            loss_sums: dict[str, float | torch.Tensor] = {}
            metric_sums: dict[str, float | torch.Tensor] = {}
            for accum_iter in range(grad_accum):
                batch = next(data_stream)
                memory_probe.snap("P1_inputs_done")
                memory_probe.snap("P2_fwd_start")
                with _maybe_save_on_cpu_ctx():
                    loss_map, outputs, step_metrics = (
                        method.single_train_step(
                            batch,
                            step,
                        ))
                    memory_probe.snap("P4_fwd_end")

                    method.backward(
                        loss_map,
                        outputs,
                        grad_accum_rounds=grad_accum,
                    )
                    memory_probe.snap("P5_bwd_peak")
                memory_probe.snap("P6_post_bwd")

                for k, v in loss_map.items():
                    if isinstance(v, torch.Tensor):
                        prev = loss_sums.get(k, 0.0)
                        loss_sums[k] = prev + v.detach()
                for k, v in step_metrics.items():
                    if k in loss_sums:
                        raise ValueError(f"Metric key {k!r} collides "
                                         "with loss key. Use a "
                                         "different name (e.g. prefix "
                                         "with 'train/').")
                    prev = metric_sums.get(k, 0.0)
                    metric_sums[k] = (prev + _coerce_log_scalar(
                        v,
                        where=("method.single_train_step()"
                               f".metrics[{k!r}]"),
                    ))

            self.callbacks.on_before_optimizer_step(
                method,
                iteration=step,
            )
            method.optimizers_schedulers_step(step)
            memory_probe.snap("P7_opt_done")
            method.optimizers_zero_grad(step)
            memory_probe.snap("P0_next")
            memory_probe.step_end()

            # Single CPU sync point: materialise GPU tensors
            # to float right before logging.
            metrics = {k: float(v) / grad_accum for k, v in loss_sums.items()}
            metrics.update({k: float(v) / grad_accum for k, v in metric_sums.items()})
            metrics["step_time_sec"] = (time.perf_counter() - t0)
            metrics["vsa_sparsity"] = float(tc.vsa_sparsity)
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)
                # Console echo so loss is visible without wandb.
                _msg_parts = []
                for _k in ("loss", "step_time_sec"):
                    if _k in metrics:
                        _msg_parts.append(f"{_k}={metrics[_k]:.6f}")
                for _k, _v in metrics.items():
                    if _k in ("loss", "step_time_sec"):
                        continue
                    _msg_parts.append(f"{_k}={_v}")
                progress.write(f"[step {step}] " + " ".join(_msg_parts))

            self.callbacks.on_training_step_end(
                method,
                metrics,
                iteration=step,
            )

            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(step)

            self.callbacks.on_validation_begin(
                method,
                iteration=step,
            )
            self.callbacks.on_validation_end(
                method,
                iteration=step,
            )

        self.callbacks.on_train_end(
            method,
            iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()

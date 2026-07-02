# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        OptimizerConfig,
        TrainingLoopConfig,
    )


class AdamWInBackward(torch.optim.Optimizer):
    """AdamW that consumes each sharded grad from a post-accumulate hook.

    This is intended for FSDP2 parameters: PyTorch calls post-accumulate hooks
    after reduce-scatter has materialized the sharded gradient, so the update
    can run immediately and clear ``p.grad`` instead of keeping every layer's
    gradient until the optimizer phase.
    """

    consumes_grad_in_backward = True

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": betas,
            "weight_decay": float(weight_decay),
            "eps": float(eps),
        }
        super().__init__(params, defaults)
        self._param_to_group: dict[int, dict] = {}
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                self._param_to_group[id(p)] = group
                self._hook_handles.append(
                    p.register_post_accumulate_grad_hook(self._make_hook(p)))

    def _make_hook(self, param: torch.nn.Parameter):

        def _hook(p: torch.nn.Parameter) -> None:
            if p is not param:
                raise RuntimeError("AdamWInBackward hook received an "
                                   "unexpected parameter")
            self._step_param(p)

        return _hook

    @torch.no_grad()
    def _step_param(self, p: torch.nn.Parameter) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("AdamWInBackward does not support sparse grads")

        group = self._param_to_group[id(p)]
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        eps = float(group["eps"])

        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, device="cpu")
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_t = state["step"]
        step_t += 1
        step = int(step_t.item())

        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)

        # Free the sharded gradient immediately; the normal optimizer step is
        # a no-op for this optimizer.
        p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("AdamWInBackward does not support closures")
        return None


class StreamingCPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW for FSDP2 ``CPUOffloadPolicy``: master weights, grads, and the
    ``(exp_avg, exp_avg_sq)`` moments live on CPU, but the update runs on the
    **GPU** via layer-by-layer streaming.

    PyTorch's default optimizer step under CPU offload executes on the host
    (slow). Here, each step streams every param's local shard (weight, grad,
    ``m``, ``v``) CPU->GPU on a dedicated H2D stream, runs the AdamW math on the
    compute stream, and streams the updated weight/``m``/``v`` back on a separate
    D2H stream into the same CPU storage FSDP all-gathers next forward. The three
    streams plus per-param CUDA events pipeline the H2D of layer i+1 under the
    compute/D2H of layer i. A fixed double-buffered GPU scratch (``_nbuf`` slots)
    bounds in-flight shards; slot reuse is gated on the prior D2H completion
    event. Numerically identical to ``torch.optim.AdamW`` (decoupled weight
    decay, same bias correction / eps).

    The update runs in the optimizer phase. The moments are managed here
    (pinned CPU) rather than seeded as DTensors, so ``manages_own_state`` tells
    ``seed_optimizer_state_for_resume`` to skip — DCP resume support is a
    follow-up. Global grad clipping still works (all grads are on CPU before
    ``step``).
    """

    manages_own_state = True

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
        pinned_state: bool = True,
        device: torch.device | None = None,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "betas": tuple(betas),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
        }
        super().__init__(params, defaults)
        self._pinned = bool(pinned_state)
        if device is not None:
            self._device = device
        elif torch.cuda.is_available():
            self._device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._device = torch.device("cpu")
        # Separate H2D and D2H streams so the next layer's prefetch (H2D) is
        # not serialized behind the current layer's compute-gated write-back
        # (D2H). A single copy stream would chain H2D(i+1) after D2H(i), and
        # since D2H(i) waits on compute(i), every stage would serialize.
        if self._device.type == "cuda":
            self._h2d_stream = torch.cuda.Stream(self._device)
            self._d2h_stream = torch.cuda.Stream(self._device)
        else:
            self._h2d_stream = None
            self._d2h_stream = None
        self._warned_unpinned = False
        # Fixed double-buffered GPU scratch (allocated lazily on first step):
        # NBUF slots × {w,g,m,v,d}, each sized to the largest local shard. Caps
        # optimizer-phase GPU reserved memory independent of layer count.
        self._nbuf = 2
        self._buffers: list[dict[str, torch.Tensor]] | None = None

    def _ensure_buffers(self) -> None:
        if self._buffers is not None or self._device.type != "cuda":
            return
        maxn = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    maxn = max(maxn, self._local(p).numel())
        self._buffers = [{
            role: torch.empty(maxn, dtype=torch.float32, device=self._device)
            for role in ("w", "g", "m", "v", "d")
        } for _ in range(self._nbuf)]

    @staticmethod
    def _local(t: torch.Tensor) -> torch.Tensor:
        """Local shard of a DTensor, else the tensor itself."""
        lt = getattr(t, "_local_tensor", None)
        return lt if lt is not None else t

    def _init_state(self, state: dict, w_local: torch.Tensor) -> None:
        state["step"] = torch.zeros((), dtype=torch.float32)
        pin = self._pinned and w_local.device.type == "cpu"
        for key in ("exp_avg", "exp_avg_sq"):
            try:
                state[key] = torch.zeros(tuple(w_local.shape),
                                         dtype=torch.float32,
                                         device="cpu",
                                         pin_memory=pin)
            except RuntimeError:  # pinning failed (e.g. host RAM) -> pageable
                if not self._warned_unpinned:
                    logger.warning("StreamingCPUOffloadAdamW: pin_memory "
                                   "failed; falling back to pageable moments "
                                   "(slower copies).")
                    self._warned_unpinned = True
                state[key] = torch.zeros(tuple(w_local.shape),
                                         dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is not None:
            raise RuntimeError("StreamingCPUOffloadAdamW does not support "
                               "closures")
        if self._h2d_stream is None:
            return self._cpu_step()
        self._ensure_buffers()

        dev = self._device
        h2d_stream = self._h2d_stream
        d2h_stream = self._d2h_stream
        compute = torch.cuda.current_stream(dev)
        nbuf = self._nbuf
        # Per-slot write-back completion events. An H2D that recycles slot s
        # must wait until the previous D2H reading out of slot s has finished,
        # otherwise the prefetch would clobber a shard still being written back.
        slot_d2h_done: list[torch.cuda.Event | None] = [None] * nbuf
        i = 0  # global param index -> round-robins the scratch slot
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_l = self._local(p)
                g_l = self._local(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, w_l)
                m_l, v_l = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = int(state["step"].item())

                # Reused fixed scratch (slot ping-pongs across nbuf buffers).
                slot_idx = i % nbuf
                slot = self._buffers[slot_idx]
                n, shp = w_l.numel(), w_l.shape
                wv = slot["w"][:n].view(shp)
                gv = slot["g"][:n].view(shp)
                mv = slot["m"][:n].view(shp)
                vv = slot["v"][:n].view(shp)
                dv = slot["d"][:n].view(shp)

                # H2D on its own stream so prefetch of layer i+1 overlaps the
                # compute/D2H of layer i. Gate slot reuse on the prior D2H out
                # of this slot (copy_ casts grad to fp32 if needed).
                if slot_d2h_done[slot_idx] is not None:
                    h2d_stream.wait_event(slot_d2h_done[slot_idx])
                with torch.cuda.stream(h2d_stream):
                    wv.copy_(w_l, non_blocking=True)
                    gv.copy_(g_l, non_blocking=True)
                    mv.copy_(m_l, non_blocking=True)
                    vv.copy_(v_l, non_blocking=True)
                h2d = h2d_stream.record_event()

                # Adam update on the compute stream after its H2D lands.
                compute.wait_event(h2d)
                with torch.cuda.stream(compute):
                    if wd != 0.0:
                        wv.mul_(1.0 - lr * wd)
                    mv.mul_(beta1).add_(gv, alpha=1.0 - beta1)
                    vv.mul_(beta2).addcmul_(gv, gv, value=1.0 - beta2)
                    bc1 = 1.0 - beta1**t
                    bc2 = 1.0 - beta2**t
                    torch.sqrt(vv, out=dv)
                    dv.div_(math.sqrt(bc2)).add_(eps)
                    wv.addcdiv_(mv, dv, value=-lr / bc1)
                comp = compute.record_event()

                # D2H write-back on its own stream, into the SAME CPU storage
                # FSDP all-gathers next forward. Runs after compute(i) lands;
                # records the slot's completion event for the next reuse.
                d2h_stream.wait_event(comp)
                with torch.cuda.stream(d2h_stream):
                    w_l.copy_(wv, non_blocking=True)
                    m_l.copy_(mv, non_blocking=True)
                    v_l.copy_(vv, non_blocking=True)
                slot_d2h_done[slot_idx] = d2h_stream.record_event()
                p.grad = None
                i += 1

        # All write-backs must finish before the next forward's all-gather.
        d2h_stream.synchronize()
        return None

    @torch.no_grad()
    def _cpu_step(self):
        """Fallback when CUDA is unavailable: plain CPU AdamW (no streaming)."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_l = self._local(p)
                g_l = self._local(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, w_l)
                m_l, v_l = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = int(state["step"].item())
                g = g_l.float() if g_l.dtype != torch.float32 else g_l
                if wd != 0.0:
                    w_l.mul_(1.0 - lr * wd)
                m_l.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v_l.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                denom = v_l.sqrt().div_(math.sqrt(bc2)).add_(eps)
                w_l.addcdiv_(m_l, denom, value=-lr / bc1)
                p.grad = None
        return None


def build_optimizer_and_scheduler(
    *,
    params: list[torch.nn.Parameter],
    optimizer_config: OptimizerConfig,
    loop_config: TrainingLoopConfig,
    learning_rate: float,
    betas: tuple[float, float],
    scheduler_name: str,
    cpu_offload: bool = False,
) -> tuple[torch.optim.Optimizer, object]:
    """Build an AdamW optimizer and LR scheduler.

    Returns ``(optimizer, lr_scheduler)`` so the caller can store them
    as method-level attributes.
    """
    if not params:
        raise ValueError("No trainable parameters passed to "
                         "build_optimizer_and_scheduler")

    use_streamed = (cpu_offload
                    and bool(getattr(optimizer_config, "streamed_offload", True))
                    and torch.cuda.is_available())
    if optimizer_config.step_in_backward:
        if int(loop_config.gradient_accumulation_steps or 1) != 1:
            raise ValueError("training.optimizer.step_in_backward requires "
                             "gradient_accumulation_steps=1")
        optimizer = AdamWInBackward(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(optimizer_config.weight_decay),
            eps=1e-8,
        )
    elif use_streamed:
        logger.info("Using StreamingCPUOffloadAdamW (GPU-executed optimizer "
                    "with layer-by-layer CPU<->GPU streaming; pinned_state=%s)",
                    bool(getattr(optimizer_config, "pinned_state", True)))
        optimizer = StreamingCPUOffloadAdamW(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(optimizer_config.weight_decay),
            eps=1e-8,
            pinned_state=bool(getattr(optimizer_config, "pinned_state", True)),
        )
    else:
        # FASTVIDEO_ADAM_FUSED=1 selects the fused AdamW kernel (in-place
        # update, no large foreach intermediate buffers). Unset -> foreach
        # default (multi-tensor), i.e. baseline behaviour unchanged.
        # capturable=True keeps the per-param `step` tensor on-device, which
        # the fused kernel requires (otherwise: "state_steps is on cpu,
        # different from other tensors on cuda").
        import os
        fused = (os.environ.get("FASTVIDEO_ADAM_FUSED", "").strip().lower()
                 in ("1", "true", "yes", "on")) or None

        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(optimizer_config.weight_decay),
            eps=1e-8,
            fused=fused,
            capturable=bool(fused),
        )

    scheduler = get_scheduler(
        str(scheduler_name),
        optimizer=optimizer,
        num_warmup_steps=int(optimizer_config.lr_warmup_steps),
        num_training_steps=int(loop_config.max_train_steps),
        num_cycles=int(optimizer_config.lr_num_cycles),
        power=float(optimizer_config.lr_power),
        min_lr_ratio=float(optimizer_config.min_lr_ratio),
        last_epoch=-1,
    )

    return optimizer, scheduler


def clip_grad_norm_if_needed(
    module: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    if max_grad_norm <= 0.0:
        return 0.0
    grad_norm = (clip_grad_norm_while_handling_failing_dtensor_cases(
        [p for p in module.parameters()],
        max_grad_norm,
        foreach=None,
    ))
    return (float(grad_norm.item()) if grad_norm is not None else 0.0)

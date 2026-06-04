# Phase 5 validation report

**Verdict:** PASS

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | `streaming_pytest_passes` | ✅ PASS | pytest exit code = 0 (expected 0; pytest.log at /workspace/FastVideo/probe_out/phase6/pytest.log) |
| 2 | `probe_full_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=0.5842348337173462 |
| 3 | `probe_full_offload_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=0.5836726427078247 |
| 4 | `probe_streamed_offload_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=0.5855268239974976 |
| 5 | `streamed_peak_at_or_below_full_offload_peak` | ✅ PASS | FULL_OFFLOAD P5_bwd_peak.max_alloc = 70.29 GB; STREAMED_OFFLOAD = 71.13 GB; slack = 1.00 GB |
| 6 | `step_peak_streamed_vs_full_recompute_info` | ℹ info | FULL P7_opt_done = 35.98 GB, STREAMED_OFFLOAD = 35.98 GB |
| 7 | `streamed_vs_full_offload_loss_match` | ✅ PASS | step 2: full_offload=0.5836726, streamed=0.5855268, diff=1.85e-03 (tol=5e-03) |
| 8 | `streamed_step_time_within_budget` | ✅ PASS | FULL_OFFLOAD step 2 = 630.45s; STREAMED_OFFLOAD step 2 = 628.22s; ratio = 1.00 (limit 1.20) |
| 9 | `streamed_filo_backward_order` | ✅ PASS | observed 40 bwd_pre events: first 3 = [39, 38, 37], last 3 = [2, 1, 0] |

## Key metrics

- FULL_OFFLOAD `P5_bwd_peak.max_alloc_gb`: 70.294352896
- STREAMED_OFFLOAD `P5_bwd_peak.max_alloc_gb`: 71.129097216
- FULL (recompute) `P7_opt_done.max_alloc_gb`: 35.982256128
- STREAMED_OFFLOAD `P7_opt_done.max_alloc_gb`: 35.982256128

## Thresholds in effect

- `--peak-slack-gb`: 1.0
- `--loss-tol`: 0.005
- `--step-time-ratio`: 1.2

Re-run with different thresholds via `python scripts/4k_milestone/phase5_verify.py --probe-root <dir> --report-md … --report-json … --peak-slack-gb X --loss-tol Y --step-time-ratio Z`.
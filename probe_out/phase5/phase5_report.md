# Phase 5 validation report

**Verdict:** FAIL

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | `streaming_pytest_passes` | ❌ FAIL | pytest exit code = 1 (expected 0; pytest.log at /workspace/FastVideo/probe_out/phase5/pytest.log) |
| 2 | `probe_full_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=4.375489234924316 |
| 3 | `probe_full_offload_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=4.372786998748779 |
| 4 | `probe_streamed_offload_completed` | ✅ PASS | exit=0, phase_memory entries=2, last loss=4.373604774475098 |
| 5 | `streamed_peak_at_or_below_full_offload_peak` | ✅ PASS | FULL_OFFLOAD P5_bwd_peak.max_alloc = 23.33 GB; STREAMED_OFFLOAD = 23.38 GB; slack = 1.00 GB |
| 6 | `step_peak_streamed_vs_full_recompute_info` | ℹ info | FULL P7_opt_done = 25.10 GB, STREAMED_OFFLOAD = 25.10 GB |
| 7 | `streamed_vs_full_offload_loss_match` | ❌ FAIL | step 2: full_offload=4.3727870, streamed=4.3736048, diff=8.18e-04 (tol=1e-05) |
| 8 | `streamed_step_time_within_budget` | ✅ PASS | FULL_OFFLOAD step 2 = 2.32s; STREAMED_OFFLOAD step 2 = 2.27s; ratio = 0.98 (limit 1.20) |
| 9 | `streamed_filo_backward_order` | ✅ PASS | observed 30 bwd_pre events: first 3 = [29, 28, 27], last 3 = [2, 1, 0] |

## Key metrics

- FULL_OFFLOAD `P5_bwd_peak.max_alloc_gb`: 23.334611968
- STREAMED_OFFLOAD `P5_bwd_peak.max_alloc_gb`: 23.37969664
- FULL (recompute) `P7_opt_done.max_alloc_gb`: 25.101680128
- STREAMED_OFFLOAD `P7_opt_done.max_alloc_gb`: 25.101680128

## Thresholds in effect

- `--peak-slack-gb`: 1.0
- `--loss-tol`: 1e-05
- `--step-time-ratio`: 1.2

Re-run with different thresholds via `python scripts/4k_milestone/phase5_verify.py --probe-root <dir> --report-md … --report-json … --peak-slack-gb X --loss-tol Y --step-time-ratio Z`.
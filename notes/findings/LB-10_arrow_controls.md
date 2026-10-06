# LB-10 Arrow controls

Implemented controls:
- `protocol_trap_control`
- `no_fake_arrow_controls`
- `evaluate_control_case` with default metric-harness integration from `metrics.default_metric_bundle`

Protocol-trap result:
- Hidden-schedule stroboscopic kernel is classified as driven (`Aff > 0`, `driven_candidate=True`).
- Phase-aware evaluation of each reversible phase gives `Aff ~= 0` and `driven_candidate=False`.

No-fake-arrow controls:
- `reversible_block_manual`
- `reversible_block_spectral`
- `null_flat_mixing_manual`
- All remain non-driven under default audits (`Aff ~= 0`, `driven_candidate=False`).

Default harness behavior:
- Behaved as intended for this ticket.
- **Did any reversible control falsely trigger driven_candidate? no**.

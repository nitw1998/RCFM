# MIMIC-AFib/mmECG ECGdeli and WFDB Bland--Altman audit

This audit applies two independent delineation implementations to the completed
RCFM-OT seeds 31, 32, and 33 on the target-informed phase-corrected central
480-sample MIMIC-AFib and mmECG outputs.

## Identity boundary

- `ecgdeli_port_fixed` is the corrected-fidelity mode of the **unofficial**
  Python port at commit `313d6dea54d88c8a616edce820c2ab9bc2b1deec`. It is not
  an execution of the official KIT-IBT MATLAB toolbox. The corresponding
  official MATLAB source is pinned at
  `c3738612771264e4d6c4686898ef7b2d6a700ad3`.
- `wfdb_ecgpuwave_1.3.4` is the traditional PhysioNet/WFDB ecgpuwave baseline.
  Its locally validated executable is recorded and hashed in each protocol.

Both methods receive identical NeuroKit2 `neurokit`-method R peaks. This holds
R detection constant and audits P/QRS/T boundary placement. Because the
phase-corrected windows are short, signals are reflect-padded to at least eight
seconds for delineation; only beats whose R peak lies in the original window
are measured.

## Aggregation and QC

The primary figures average jointly measurable windows within source record for
MIMIC-AFib (`n=34`) or subject for mmECG (`n=11`) before Bland--Altman analysis.
Each seed is retained separately. Raw figures are kept to reveal delineation
failures; primary QC figures require ordered fiducials and RR 300--2000 ms, PR
80--220 ms, QRS 60--180 ms, and QT 200--600 ms. AF-labelled MIMIC windows are
excluded from P-amplitude and PR measurements because a stable P wave is not
applicable there.

## Result synopsis

All 2,040 MIMIC and 2,494 mmECG reference/generated signal calls completed for
both algorithms and all three seeds. The generated-minus-reference QC results,
averaged descriptively across the three seed-specific biases and LoA widths,
are:

| Dataset | Method | HR bias / LoA width (bpm) | RR bias / width (ms) | PR bias / width (ms) | QRS bias / width (ms) | QT bias / width (ms) |
|---|---|---:|---:|---:|---:|---:|
| MIMIC-AFib | ECGdeli port fixed | -0.565 / 7.521 | 5.296 / 40.780 | -2.261 / 37.038 | 0.159 / 9.986 | 0.732 / 32.028 |
| MIMIC-AFib | WFDB ecgpuwave | -0.566 / 6.814 | 5.198 / 39.836 | 0.986 / 12.978 | -1.249 / 16.166 | -0.426 / 39.824 |
| mmECG | ECGdeli port fixed | 0.092 / 2.333 | -0.695 / 22.167 | -0.892 / 19.736 | -2.113 / 8.670 | 0.047 / 23.277 |
| mmECG | WFDB ecgpuwave | 0.103 / 2.600 | -0.801 / 23.523 | -0.047 / 14.285 | -0.676 / 8.016 | -2.957 / 25.186 |

These cross-seed averages summarize three separate descriptive B--A analyses;
they are not pooled LoA and do not treat seeds as clinical replicates.
ECGdeli/WFDB agreement supports the conclusion that the earlier DWT QRS scale
was delineator-dependent. P/PR remains algorithm-sensitive, especially in
MIMIC-AFib, and must retain the AF exclusion, point-order QC, and coverage report.

## Artifacts

- MIMIC-AFib: `runs/diagnostics/mimic_afib_rcfm_ot_three_seed_ecgdeli_wfdb_ba_v2/`
- mmECG: `runs/diagnostics/mmecg_rcfm_ot_three_seed_ecgdeli_wfdb_ba_v2/`

Each directory contains raw and QC timing figures, QC amplitude figures,
per-window and per-group tables, a complete B--A summary, input/output hashes,
and the executable command. The adjacent `v1` directories have identical
numbers but a cramped preliminary figure layout and are superseded by `v2`.

## Window-level companion

The group-level audit remains the primary identity-aware view. A separate
window-level companion is generated directly from its frozen per-window table,
without rerunning either delineator:

- MIMIC-AFib: `runs/diagnostics/mimic_afib_rcfm_ot_three_seed_ecgdeli_wfdb_window_ba_v1/`
- mmECG: `runs/diagnostics/mmecg_rcfm_ot_three_seed_ecgdeli_wfdb_window_ba_v1/`

Every algorithm has a three-row timing panel (one row per seed) in raw and QC
form and a three-row QC amplitude panel. The window-level figures intentionally
retain the sample-quantized diagonal/diamond structures. Those patterns follow
from discrete fiducial indices and repeated beat averages and are not removed by
plot smoothing. LoA are much wider than group-level LoA because they include
within-record/within-subject window variability. Overlapping windows and shared
identities are correlated, so these plots are descriptive morphology audits and
must not be used as independent-replicate clinical inference.

This is an automated algorithmic sensitivity audit, not expert fiducial ground
truth, clinical equivalence evidence, or a deployable phase-alignment protocol.

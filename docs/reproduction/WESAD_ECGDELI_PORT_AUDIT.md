# WESAD ECGdeli port sensitivity audit

## Implementation identity

The official KIT-IBT ECGdeli is a MATLAB toolbox and depends on the MATLAB
Image Processing, Signal Processing, Statistics and Machine Learning, and
Wavelet toolboxes. MATLAB/Octave is not installed in the experiment
environment, so the official implementation was not executed.

This audit instead uses the explicitly unofficial community Python port at
commit `313d6dea54d88c8a616edce820c2ab9bc2b1deec`. Its documented upstream is
the official MATLAB commit `c3738612771264e4d6c4686898ef7b2d6a700ad3`. The
port describes itself as independent, not endorsed by KIT-IBT, and primarily
AI-generated under human direction. Results must therefore be labelled
`ECGdeli (unofficial Python port)` and cannot establish official MATLAB
behaviour.

## Runtime

Set `RCFM_WORKSPACE` to the coordination-workspace root and `ECGDELI_PYTHON`
to a Python 3.11-or-later interpreter. The audit uses isolated dependencies
under `$RCFM_WORKSPACE/runs/tools/ecgdeli-py-deps`. The bundled
115,200-sample, 12-lead, 1000-Hz example completes with 197 beats before any
WESAD experiment is accepted.

Run the corrected-fidelity sensitivity experiment with:

```bash
PYTHONPATH="$RCFM_WORKSPACE/runs/tools/ecgdeli-py-deps:$RCFM_WORKSPACE/external/ecgdeli-py-313d6dea/src" \
"$ECGDELI_PYTHON" \
  scripts/audit_wesad_ecgdeli_port_delineation.py \
  --phase_npz "$RCFM_WORKSPACE/runs/evaluation/wesad_random_window_record_minmax_rcfm_ot_e200_s31_phase_v1/phase_predictions_maxlag16.npz" \
  --previous_fiducials "$RCFM_WORKSPACE/runs/diagnostics/wesad_rcfm_ot_s31_delineation_audit_v1/per_beat_fiducials.csv" \
  --ecgpuwave_fiducials "$RCFM_WORKSPACE/runs/diagnostics/wesad_rcfm_ot_s31_ecgpuwave_audit_v3/per_beat_ecgpuwave_fiducials.csv" \
  --output_dir "$RCFM_WORKSPACE/runs/diagnostics/wesad_rcfm_ot_s31_ecgdeli_port_fixed_v3" \
  --fidelity fixed
```

The same prior NeuroKit R peaks are supplied through ECGdeli's reference FPT;
ECGdeli then re-checks the QRS locally. ECGdeli's documented filter chain
(1--40 Hz band-pass, 50-Hz notch, isoline correction) is used because its
annotator expects a filtered signal. This is therefore a complete ECGdeli
pipeline sensitivity baseline, not a boundary-only comparison on the exact
NeuroKit-cleaned samples.

## Results and boundary

The port's `matlab` fidelity mode deliberately reproduces upstream defects and
fails on 72 of the 400 reference/generated window calls, predominantly with a
MATLAB-style array-bounds exception. Its `fixed` mode succeeds for every
generated window and 199/200 reference windows; the one remaining reference
window has fewer than the required three supplied R peaks.

In `fixed` mode, QRS onset--R--offset ordering is valid for every returned beat.
Reference/generated median QRS is 125/125 ms, and 98.58%/99.33% of measurable
QRS intervals lie in the audit range 60--180 ms. The extreme DWT QRS cases of
335.9/333.3 ms become ECGdeli window means of 111.3/109.4 ms. Matched QRS is
46.875 ms wider than NeuroKit prominence and 23.4375/7.8125 ms narrower than
ecgpuwave for reference/generated signals.

P delineation remains unreliable without explicit QC. Only 77.57%/82.45% of
returned reference/generated beats have valid P-onset--peak--offset--QRS
ordering. Some invalid rows put P onset 1.6--2.1 seconds before QRS, causing the
raw window-level PR Bland--Altman SD to reach 597 ms. Requiring valid P order
and PR in 80--220 ms leaves 159/200 paired windows; the resulting PR bias is
0.808 ms with SD 27.528 ms. This filtered result describes the selected valid
subset and must not be confused with unconditional performance.

The authoritative artifact is
`runs/diagnostics/wesad_rcfm_ot_s31_ecgdeli_port_fixed_v3/`. It contains raw
and QC window tables/plots so the effect of exclusion is visible. No manuscript
or clinical evidence table should use the port until official MATLAB execution
or expert-fiducial validation is available.

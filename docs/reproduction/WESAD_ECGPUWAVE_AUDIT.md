# WESAD ecgpuwave delineation audit

This diagnostic tests PhysioNet ecgpuwave as an independent traditional
P/QRS/T boundary locator for the 128-Hz WESAD random-window evaluation. It is
not the authoritative clinical evaluator and does not treat algorithm output
as expert ground truth.

## Fixed sources

- WFDB 10.7.0: `https://github.com/bemoody/wfdb.git`, tag `10.7.0`, commit
  `de6b1d3981d69060e6a4d1b4e86375fb0d3ed2d1`.
- ecgpuwave 1.3.4:
  `https://physionet.org/physiotools/ecgpuwave/src/ecgpuwave-1.3.4.tar.gz`,
  SHA-256
  `ba73dc315350bda22d57f1632c4bc676be1eb70bc7f76c469468517de993ab9d`.

Both packages are GPL-licensed. Source checkouts/archives belong under the
coordination workspace's `external/` directory and build products under
`runs/tools/`; neither belongs in the Git repository.

## Local build

Set `RCFM_WORKSPACE` to the coordination-workspace root before running these
commands.

WFDB was configured for local records only because NETFILES, FLAC, and WAVE
are unnecessary for this audit:

```bash
cd external/wfdb-10.7.0
./configure \
  --prefix="$RCFM_WORKSPACE/runs/tools/wfdb-10.7.0" \
  --without-netfiles --without-flac --without-xview
make install

cd ../ecgpuwave-1.3.4
PATH="$RCFM_WORKSPACE/runs/tools/wfdb-10.7.0/bin:/usr/bin:/bin" make
PATH="$RCFM_WORKSPACE/runs/tools/wfdb-10.7.0/bin:/usr/bin:/bin" make check
PATH="$RCFM_WORKSPACE/runs/tools/wfdb-10.7.0/bin:/usr/bin:/bin" \
  make install prefix="$RCFM_WORKSPACE/runs/tools/ecgpuwave-1.3.4"
```

The byte-identical ecgpuwave check differs because of documented
floating-point variation; the official second check, which permits at most
one sample of boundary variation, passes.

## Audit command

```bash
python scripts/audit_wesad_ecgpuwave_delineation.py \
  --phase_npz "$RCFM_WORKSPACE/runs/evaluation/wesad_random_window_record_minmax_rcfm_ot_e200_s31_phase_v1/phase_predictions_maxlag16.npz" \
  --previous_fiducials "$RCFM_WORKSPACE/runs/diagnostics/wesad_rcfm_ot_s31_delineation_audit_v1/per_beat_fiducials.csv" \
  --ecgpuwave "$RCFM_WORKSPACE/runs/tools/ecgpuwave-1.3.4/bin/ecgpuwave" \
  --output_dir "$RCFM_WORKSPACE/runs/diagnostics/wesad_rcfm_ot_s31_ecgpuwave_audit_v3"
```

The entry point supplies the same NeuroKit `neurokit`-method R peaks through
ecgpuwave's `-i` option. Therefore the comparison tests waveform-boundary
localization rather than changing the R detector. Each signal is cleaned with
the existing NeuroKit method, reflect-padded by 16 samples per side, encoded as
a temporary WFDB record, delineated, parsed, and discarded. Only R peaks in
the original 480-sample support are retained.

## Result and boundary

On the same deterministic 200-window sensitivity subset, ecgpuwave produces
valid QRS onset--R--offset order for 100% of returned beats and valid P ordering
for 95.90%/95.80% of reference/generated beats. Median PR is 140.625 ms for
both; median QRS is 132.8125/117.1875 ms. It removes the visually confirmed
DWT failure in which QRS offset was placed near the T wave in the extreme
333--336 ms cases.

The result is not sufficient to replace the evaluator. ecgpuwave returns
707/667 reference/generated beats versus 919/889 for prominence on this short
window subset, predominantly because it does not return the final boundary
beat without later context. Its QRS duration is also systematically wider
than prominence by a matched-beat median of 62.5/54.6875 ms, and 26.17%/18.59%
of measurable reference/generated QRS intervals exceed the audit threshold of
180 ms. Noisy windows can still receive complete-looking annotations.

Use the output as an algorithm-sensitivity audit only. A formal replacement
requires a prespecified quality policy, continuous/context-preserving input,
and validation against expert fiducials such as LUDB or QTDB.

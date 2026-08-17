# Independent CATransformer reproduction

## Status and claim boundary

This repository labels the implementation `CAT-PPG (reproduced)`. It is an
independent paper-based reproduction of Yuan et al., *CATransformer: A
Cycle-Aware Transformer for High-Fidelity ECG Generation From PPG*, DOI
`10.1109/JBHI.2024.3482853`; it is not an official implementation. A GitHub
repository search for `CATransformer PPG ECG` on 2026-08-11 returned no
repositories. A DOI-based GitHub code search could not be completed because
that API endpoint required authentication. These search results do not prove
that no author implementation exists.

The executable experiments apply the architecture to frozen local MIMIC-AFib
and WESAD PPG-to-ECG artifacts. They do not reproduce the paper's pooled
five-data set, participant-level 80/20 split, five-fold protocol, or reported
table values. PTB-XL and CPSC2018 use a separately labelled `CAT-ECG (adapted)`
entry: Lead II alone drives source-only cycle extraction, and an explicit
learned pointwise head maps the paper's single-channel CAT output to the other
11 leads. That head is `adaptation-required`, is not paper-explicit, and must
not be described as part of the official or reproduced CAT architecture. A
shared CAT backbone plus this output head, rather than 11 independent CAT
networks, is an `author-choice` that keeps the comparator's resource cost
within the same order as the original single-output model.

MMECG is separately labelled `CAT-RCG (adapted)`. It retains the paper's
single-input/single-output architecture but substitutes energy-weighted RCG
for PPG; dominant cycles are extracted from RCG only. This modality change is
`adaptation-required`, not a reproduction of the paper's PPG experiment.

## Paper-to-code traceability

| Component | Paper location and statement | Local implementation | Shape or setting | Certainty | Verification |
|---|---|---|---|---|---|
| Input protocol | Sec. IV-A: filtered simultaneous PPG/ECG, 128 Hz, z-score, min-max to `[-1,1]`, 4-second windows | `cat_data.load_mimic_cat_datasets` | `(B,1,512)` source and target | `adaptation-required` | Frozen manifest, size and all-zero checks |
| Source cycle spectrum | Sec. III-A, Eq. (1): `Amp(FFT(X))` | `SourceCycleExtractor` | real FFT, DC excluded | `paper-explicit` | Known-sinusoid and source-only tests |
| Dominant cycles | Sec. III-A, Eq. (2): top-k frequencies and cycle lengths; Sec. IV-A sets `k=2` | `SourceCycleExtractor` | `top_k=2`, ceiling length used to realize the paper's zero-padded reshape | `paper-explicit` with boundary interpretation | Known-cycle, zero-source fallback tests |
| Period views | Sec. III-A, Eq. (3): reshape padded source to `c_i x f_i` | `CycleViewBuilder` | one view per selected frequency | `paper-explicit` | Equation-view and variable-prefix tests |
| Fixed patch width | Sec. III-A, Eq. (4): last-value padding or truncation to width `T` | `CycleViewBuilder` | `T=64` | `author-choice` for the unreported value of `T` | Exact padding/truncation fixture |
| Embedding | Sec. III-B, Eq. (5) | `CycleAwareTransformerBlock.embedding` | `Linear(64,128)` plus sinusoidal positions | `paper-inferred` dimensions/positions | Output and gradient tests |
| Transformer | Sec. III-B, Eq. (6); Sec. IV-A: two CAT layers, four encoder layers per CAT layer | `CycleAwareTransformerBlock.encoder`, `CATransformer.blocks` | 2 blocks, 4 encoders/block, 4 heads, FFN 512 | `paper-explicit` depth; `author-choice` width, heads, FFN, dropout | Config freeze and architecture-change rejection |
| Reconstruction | Sec. III-B, Eq. (7): flatten and truncate to the original length | `CycleAwareTransformerBlock._restore` | `(B,1,512)`; deterministic tail fill is logged only if encoded support is insufficient | `paper-explicit` plus defensive fallback | Flatten/truncate/tail test |
| Aggregation | Sec. III-C, Eqs. (8)-(9): softmax FFT amplitudes and weighted branch sum | `CycleAwareTransformerBlock.forward` | two weights per record | `paper-explicit` | Determinism and diagnostic tests |
| Repeated block/output | Algorithm 1 and Sec. III-C: repeat the block twice and return the second output | `CATransformer.forward` | single ECG channel, no extra output projection | `paper-explicit` | State-dict guard rejects an added projection; multi-lead output rejected |
| Objective | Sec. III-C, Eq. (10): MSE plus KL divergence | `CATLoss` | softmax over 512 time samples, temperature 1, KL weight 1 | `paper-explicit` objective; `paper-inferred` probability axis; `author-choice` temperature/weight | Finite backward test |
| Optimization | Sec. IV-A: Adam, batch 128, learning rate `1e-4` | `scripts/train_cat.py` and frozen config | 500 epochs, seed 31, AMP, gradient clip 1 | `paper-explicit` optimizer/batch/LR; `author-choice` epochs/seed/AMP/clip | CPU and GPU batch-128 smoke |
| Inference | Algorithm 1 is a deterministic forward map | `CATransformer.forward`, `scripts/evaluate_cat.py` | `NFE=1`; cycle extraction accepts source PPG only | `paper-explicit` | Target-argument leakage guard and repeated-forward test |

## Frozen local experiment

The local input is
`runs/preprocessing/mimic_afib_rddm_zero_ppg_qc_v1/MIMIC-AFib/`:

- 8,400 train and 1,800 final-test paired windows, four seconds at 128 Hz;
- split membership SHA-256
  `a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51`;
- independent per-window min-max normalization to `[-1,1]` at load time;
- no all-zero PPG windows;
- array-row pairing only, with no additional phase alignment;
- unavailable subject identifiers, so subject-disjointness is not verified.

This differs materially from the paper's training corpus and preprocessing. In
particular, the local artifact does not provide the participant-specific
z-score state needed to reproduce the paper's stated z-score-then-min-max
pipeline. Results from this entry therefore test the reproduced architecture
under the existing MIMIC-AFib comparison protocol, not the paper's exact data
pipeline.

The held-out 1,800 windows are never evaluated by the trainer. Formal
evaluation is allowed only for the predeclared epoch-500 checkpoint unless
`--allow_nonfinal_checkpoint` is explicitly used for a smoke test.

## Checkpoint and outage recovery

Every completed epoch atomically replaces `checkpoint_latest.pt`; every 25th
epoch also writes `checkpoint_epoch_N.pt`. A checkpoint contains model,
optimizer, AMP scaler, Python/NumPy/Torch RNG, DataLoader generator state,
resolved config, normalization, output shape and provenance. Resume creates a
new run directory and rejects changes to data, architecture, optimization,
batch size, worker count or seed. The final epoch may be extended, while
logging/checkpoint frequency and output paths remain operational settings.
Finite-loss AMP gradient overflow is recorded as `amp_overflow_rate`; the
optimizer update is skipped and `GradScaler` lowers its scale before training
continues. A nonfinite forward loss, or nonfinite gradients without AMP,
remains a fatal error.

The isolated workspace launcher is `scripts/launch_cat_ppg_mimic_gpu5.sh`.
Run it from the coordination workspace, not from the linked repository. For a
fresh run:

```bash
nohup bash scripts/launch_cat_ppg_mimic_gpu5.sh \
  > /tmp/cat_ppg_mimic_launcher.log 2>&1 &
```

For recovery, point `RCFM_CAT_RESUME` at a validated CAT checkpoint. The
launcher always creates a new run ID and never overwrites the interrupted run:

```bash
RCFM_CAT_RESUME=/absolute/path/checkpoint_latest.pt \
nohup bash scripts/launch_cat_ppg_mimic_gpu5.sh \
  > /tmp/cat_ppg_mimic_resume_launcher.log 2>&1 &
```

`RCFM_CAT_GPU` defaults to physical GPU 5 and `RCFM_CAT_RUN_ID` can set the new
run ID. W&B mode is online in the frozen config. Logs, PID files, checkpoints
and W&B local files remain under
`runs/training/cat_ppg_mimic_afib_v1/`.

## Verification

The synthetic suite covers source-only extraction, known and failed spectra,
variable valid lengths, exact view construction, gradient preservation,
reconstruction, deterministic shape, leakage guards, objective backward,
checkpoint round-trip and resume-contract mismatch rejection. Real-data smoke
on 2026-08-11 completed epoch-boundary resume and produced a model bitwise
identical to uninterrupted CPU training. A full-structure batch-128 AMP step
and a subsequent CUDA checkpoint resume both completed on a 48 GiB Quadro RTX
8000. Smoke artifacts are disposable under `/tmp` and are not experiment
results.

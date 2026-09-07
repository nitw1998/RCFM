# FACM external adaptation for RCFM-OneStep deployment

## Status and provenance

This implementation treats Flow-Anchored Consistency Models (FACM) as an
external deployment-optimization method, not as an original contribution of
RCFM. The source is the official `ali-vilab/FACM` repository, inspected at
commit `8d80d4c65101f814095984a91329ce4aa37be79b`. The checkout is kept outside
this repository. Its license is Apache-2.0. The relevant upstream files are
`losses.py`, `sampler.py`, `train.py`, and `scripts/{train,test}.sh`.

Official ImageNet usage initializes distillation from `800ep-stg1.pt`, trains
with `FACMLoss`, and invokes `consistency_model_sampler` with one or two steps.
The official commands are `bash scripts/train.sh`, `bash scripts/test.sh
--ckpt-path cache/400ep-stg2.pt --sampling-steps 1`, and `bash scripts/test.sh
--ckpt-path cache/100ep-stg2.pt --sampling-steps 2`. The upstream checkout has
no locked requirements file; its executable stack imports PyTorch,
`accelerate`, NumPy, SciPy, Lightning-DiT modules, a VAE, and the separately
installed `torch-fidelity` fork. The local adapter deliberately ports none of
the image loader, VAE/latent, FID, DiT, or class-label CFG stack. It is tested
with the repository environment's PyTorch 2.0.0/CUDA 11.7 installation.
An official checkpoint smoke test was not part of this physiological-signal
adaptation because the ImageNet VAE, latent statistics, data, and released checkpoint
assets are not installed locally. No ImageNet reproduction or FID claim is
made.

## Frozen base models

The current retraining entry uses completed seed-31, epoch-200 canonical CFM
checkpoints whose reference sampler was evaluated with NFE=50. FACM consumes
the frozen CFM velocity field directly at sampled times; it does **not** run a
50-step ODE inside each FACM update. Each CFM has `region_weight=0`, no
minibatch OT, and the matched random-window 80/20 data protocol. The new
configs pin checkpoint kind, epoch, reference NFE, SHA-256, split,
normalization, and channel contract:

| Config | Task | Train/validation | Target channels |
|---|---|---:|---:|
| `facm_cfm50_ptbxl.yaml` | Lead II to other ECG leads | 34,939 / 8,735 | 11 |
| `facm_cfm50_cpsc2018.yaml` | Lead II to other ECG leads | 19,364 / 4,842 | 11 |
| `facm_cfm50_mimic.yaml` | PPG to ECG | 8,160 / 2,040 | 1 |
| `facm_cfm50_wesad.yaml` | PPG to ECG | 17,365 / 4,342 | 1 |
| `facm_cfm50_mmecg.yaml` | RCG to ECG | 9,973 / 2,494 | 1 |

The older `facm_{dataset}.yaml` files and their validator profile remain for
reproducing the already completed epoch-500 RCFM-teacher experiment. They are
not selected by the current aggregate launcher.

### Legacy RCFM-teacher profile

FACM initialization and the frozen training-only teacher use completed seed-31,
epoch-500 canonical RCFM checkpoints. All five teachers use the linear
noise-at-0/data-at-1 path, region weight 0.01 during original RCFM training,
no minibatch OT, and 50-NFE reference inference. FACM does not consume the
original region mask or an OT solver. The five configs pin the exact teacher
SHA-256, split, normalization and channel contract:

| Config | Task | Train/held-out | Target channels |
|---|---|---:|---:|
| `facm_ptbxl.yaml` | Lead II to other ECG leads | 17,440 / 2,193 | 11 |
| `facm_cpsc2018.yaml` | Lead II to other ECG leads | 5,487 / 686 | 11 |
| `facm_mimic.yaml` | PPG to ECG | 8,400 / 1,800 | 1 |
| `facm_wesad.yaml` | PPG to ECG | 17,494 / 4,213 | 1 |
| `facm_mmecg.yaml` | RCG to ECG | 9,590 / 2,877 | 1 |

## Objective mapping

| FACM component | Official implementation | Conditional 1-D adaptation |
|---|---|---|
| Signal shape | `(B,C,H,W)` image latent | `(B,C,T)` ECG |
| Interpolant | `x_t=t*x_1+(1-t)*x_0` | unchanged |
| Teacher velocity | frozen pretrained FM at `t` | frozen audited CFM/legacy RCFM flow and condition encoder at `t` |
| FM task condition | expanded time `2-t` | unchanged; existing RCFM time embedding accepts `[1,2]` |
| CM task condition | time `t` | unchanged |
| JVP tangents | `(teacher velocity, 1)` | unchanged, via `torch.func.jvp` |
| CM relaxation | `alpha=1-t^p`, `p=0.5`, clipped residual | unchanged |
| CM weighting | `cos(pi*t/2)` and norm-L2 | unchanged |
| FM anchor | MSE plus cosine | MSE plus full `(C,T)` vector cosine |
| Conditioning | ImageNet label and CFG | PPG feature pyramid; no CFG/null PPG branch exists |
| Combined loss | `L_CM + L_FM` | unchanged, equal weights |

The full-signal cosine is an explicit 1-D adaptation of Eq. 10. The upstream
code computes cosine only over the image channel dimension; for single-lead
ECG that would degenerate to a pointwise sign comparison rather than a signal
cosine. No region-weighted FACM anchor is used in the principal port.

The local training microbatch is 4 with 32-step gradient accumulation
(effective batch 128). This is an RCFM hardware adaptation and not an upstream
FACM default. Training remains FP32 until the required JVP numerical and memory
gate passes; AMP is not silently enabled.

## One-step inference contract

For the local convention, one-step inference is exactly

```text
condition_features = student_condition(source_signal) # one encoder call
x_1 = z + student_flow(z, condition_features, t=0)   # one student call / NFE=1
```

The one-step interface cannot accept the teacher, target ECG, training region
mask, Grad-CAM model, or OT solver. Student checkpoints intentionally exclude
all teacher parameters. A later evaluator must count the student call and
condition-encoder call and report matched latency/memory; those measurements
must not be inferred from this training implementation.

## Aggregate training entry

The public method name is `RCFM-OneStep`; FACM is recorded as the external
training method/provenance. The aggregate launcher now selects the CFM-NFE50
profile. To start one dataset and one seed on GPU 0:

```bash
bash repo/scripts/launch_rcfm_onestep_five_dataset.sh 0 ptbxl 31
```

To queue all five datasets and seeds 31/32/33 sequentially on one idle GPU:

```bash
bash repo/scripts/launch_rcfm_onestep_five_dataset.sh 0 all all
```

Run these commands from the coordination workspace root. The launcher defaults
to the audited workspace preprocessing artifacts and
CFM checkpoints. Dataset roots can be overridden with `PTBXL_DATA_ROOT`,
`CPSC2018_DATA_ROOT`, `MIMIC_AFIB_DATA_ROOT`, `WESAD_DATA_ROOT`, and
`MMECG_DATA_ROOT`; checkpoint paths have corresponding
`RCFM_ONESTEP_<DATASET>_CFM_CHECKPOINT` variables. `RCFM_RUNS_ROOT`,
`RCFM_WORKSPACE_ROOT`, `RCFM_PYTHON`, and `WANDB_MODE` are also configurable.
Set `RCFM_WORKSPACE_ROOT` explicitly when invoking the script from the physical
Git worktree rather than through the coordination workspace. The queue runs in the
background and prints its PID, log, and result root.

For a direct foreground MIMIC invocation, set local paths without adding them
to the config:

Set local paths without adding them to the config:

```bash
export RCFM_DATA_ROOT=/path/to/data
export RCFM_RUNS_ROOT=/path/to/runs
export RCFM_FACM_TEACHER_CHECKPOINT=/path/to/cfm_checkpoint_epoch_200.pt
python scripts/train_facm_acceleration.py \
  --config configs/one_step/facm_cfm50_mimic.yaml \
  --device cuda:0
```

For a disposable numerical/JVP smoke gate, add `--epochs 1 --save_every 1
--max_train_records 4 --max_heldout_records 4 --max_batches 1 --run_id ...`.
The run directory records the base checkpoint hash, repository state,
environment, complete FACM contract, loss components, peak allocated GPU
memory, and checkpoints. Formal training must not begin until an idle GPU is
available and the one-batch FP32 JVP smoke gate succeeds.

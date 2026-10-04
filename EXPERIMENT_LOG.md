# AlloyFlow: Experiment Log

This log tracks every controlled experiment and ablation run in **AlloyFlow** on the 6-DoF **SO-ARM101** multi-task benchmark (`Task 0: pick_pen_holder_to_bowl`, `Task 1: pick_cup_to_bowl`, `Task 2: stack_cup_on_cube`). Each entry records what we tested, the exact dataset and training configuration used, closed-loop evaluation metrics across training and held-out test splits, and what we learned in plain English.

***

## 1. Master Results Table

### 1.1 Mode 1 (`Sim-Only`) Iterative Ablation Log

| Run ID | Configuration Summary | Dataset (`300` Sim Demos) | Train Split Pass (`60` eps) | Held-Out Test Pass (`60` eps) | Reached Source `< 2.5 cm` (`Train / Test`) | Best `loss` / `val_ode_mse` | Main Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Proto 1a..1e** | Initial architecture debugging (time embedding, `8x8` learnable `SpatialSoftmax2d`, 2-layer `obs_proj`, frame trimming) | `data/sim_demos.h5` (`150` Clean + `150` DR) | `0 / 60` | `0 / 60` (`4 / 8` placed in bowl on 12-seed check before release fix) | `N/A` (3-demo visual checks) | `0.0469` to `0.1264` | Fixed vision feature collapse and truncated release/retract frames; removed ad-hoc time overrides |
| **Exp 01 (`exp01_clean_lags32`)** | Clean 8-segment dataset + multi-scale history `lags=(0,4,8,16,32)` (`30D`) + `max_steps=280` | `data/sim_demos.h5` (`43,131` steps) | `0 / 60` (`0.0%`) | **`1 / 60` (`1.7%`)** (`test_t0_s9013`) | `24 / 60` (`40.0%`) / `15 / 60` (`25.0%`) | `0.07355` / `0.00140` | Full pick-place-release-retract works end-to-end once aligned (`test_t0_s9013`), but `30D` velocity history causes a severe copycat shortcut (`Step 00` swap ratio `0.10`, `overhead_cam` $R^2 \le 0.00$) |
| **Exp 02a (`exp02a_sharp_nohist`)** | Sharp `16x16` encoder (`conv4` stride 1, no pre-softmax `SiLU`, `temp=0.1`) + single-frame proprio `lags=(0,)` (`6D`) | `data/sim_demos.h5` (`43,131` steps) | `0 / 60` (`0.0%`) | **`1 / 60` (`1.7%`)** (`test_t1_s19000`, `6 / 60` lifted `> 2 cm` vs `2 / 60`) | `22 / 60` (`36.7%`) / `16 / 60` (`26.7%`) | `0.07787` / `0.00198` | Step-0 camera-swap ratio rose **`3.4x`** (`0.10 -> 0.34`), `t=90` swap ratio reached **`0.78`**, `overhead_cam` pan $R^2$ jumped from `<= 0.00` to **`+0.66`**, and test lifts tripled (`2/60 -> 6/60`), but 300 demos under-constrain radial depth (`shoulder` $R^2 = +0.02$) |
| **Exp 02b (`exp02b_pretrain_loc`)** | Pretrain camera encoders on rendered labeled layouts + training-only position head | `data/sim_demos.h5` + rendered layouts | `Planned` | `Planned` | `Planned` | `Planned` | Close the remaining localization gap across all 4 objects (`pan` + `shoulder`) |

### 1.2 4-Mode Benchmark Comparison Table (Target Matrix)

| Mode | Training Mode Flag | Dataset Used | Sim Success (`T0 / T1 / T2 / Avg`) | Real Success (`T0 / T1 / T2 / Avg`) | Step Latency | Main Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Mode 1** | `sim_only` | `100` Sim demos/task (`50 Clean + 50 DR`, `300` total) | `0.0% / 0.0% / 0.0% / 0.0%` (`Train`), `0.0% / 5.0% / 0.0% / 1.7%` (`Test`, `Exp 02a`) | `TBD` (Zero-Shot) | `~13 ms` | Baseline Sim accuracy and zero-shot gap on real robot |
| **Mode 2** | `real_only` | `20` Real demos/task (`60` total) | `Not run` | `TBD` | `TBD` | How well the real robot learns from small real data alone |
| **Mode 3** | `finetune` | Start from `Mode 1`, finetune on `60` Real demos | `TBD` | `TBD` | `TBD` | Does finetuning improve Real accuracy or forget Sim skills? |
| **Mode 4** | `cotrain` | `50%` Sim (`300` demos) + `50%` Real (`60` demos) per batch | `TBD` | `TBD` | `TBD` | Does mixing Sim and Real in every batch work best overall? |

***

## 2. Prototype Debugging Summary (`Proto 1a` to `Proto 1e`, Pre-Clean Baseline)

Before locking the clean benchmark in **Experiment 01**, five diagnostic runs on `data/sim_demos.h5` uncovered and fixed four architectural bugs and one dataset truncation bug (preserved on branch `snapshot/pre-clean-eval`):
1. **Removed High-Frequency Flow Time Aliasing ([`SinusoidalTimeEmbedding`](./training/model.py#L223-L236))**: Removed the `* 1000.0` multiplier on continuous flow time $\tau \in [0, 1]$, which previously caused `159` full $2\pi$ oscillations across the `[0, 1]` ODE interval (`100 rad` phase jumps per $\Delta\tau = 0.1$ Euler step).
2. **Prevented Spatial Softmax Background Dilution ([`SpatialSoftmaxConvNet`](./training/model.py#L76-L152))**: Changed `conv4` to `stride = 2` (`8x8 = 64` spatial cells instead of `16x16 = 256` cells) and made the softmax temperature in [`SpatialSoftmax2d`](./training/model.py#L16-L74) a learnable `nn.Parameter`.
3. **Restored Camera Token Scale & Nonlinear Conditioning ([`MultiCameraAttention`](./training/model.py#L154-L220), [`TaskConditionedVisionFlowPolicy`](./training/model.py#L309-L315))**: Scaled attention weights by `n_cams` (`3 * attn_weights * v`) with a per-slot residual connection and `LayerNorm(32)`, and upgraded `obs_proj` from a single linear layer to a 2-layer nonlinear MLP (`Linear -> LayerNorm -> SiLU -> Linear`).
4. **Fixed Truncated Release & Retract Frames ([`filter_episode_frames`](./training/dataset.py#L40-L79))**: Early frame-trimming heuristics accidentally deleted the post-placement gripper opening (`release_lag`) and upward retraction (`Segment 8`) frames, causing rollouts that placed the object inside the bowl (`0.43 cm` to `1.71 cm` from center) to fail [`SimEnv.check_task_success`](./envs/sim_env.py#L461-L479) because the gripper never opened and retracted clear of the object.

***

## 3. Experiment 01: Clean Baseline (`exp01_clean_lags32`)

* **Commit / Branch**: `fc826a0` on `main` (and `exp/clean-eval-and-data`; pre-clean snapshot preserved on `snapshot/pre-clean-eval`)
* **Checkpoint**: `checkpoints/exp01_clean_lags32/best_policy.pt`
* **Benchmark JSON**: `checkpoints/exp01_clean_lags32/eval_benchmark_sim_clean.json`
* **Diagnostic GIFs**: `checkpoints/exp01_clean_lags32/gifs/` (`demo_0000_task0.gif`, `demo_0101_task1.gif`, `demo_0200_task2.gif`, `test_t0_s9013.gif`)
* **Dataset**: `data/sim_demos.h5` (`300` expert trajectories: `100` per task, `50` `sim_clean` + `50` `sim_dr`, `144` steps per episode at `20 Hz`)

### 3.1 What Changed vs. Initial Prototype (`snapshot/pre-clean-eval`)

1. **Restored Complete 8-Segment Transitions ([`training/dataset.py`](./training/dataset.py#L40-L79))**:
   - Removed aggressive frame dropping (`clamp_dwell`, `release_lag`, `retract_bridge`, and terminal truncation) inside [`filter_episode_frames`](./training/dataset.py#L40-L79) so that the policy trains on the complete `8`-segment trajectory (`approach -> descend -> clamp -> lift -> transit -> lower -> release -> retract`).
   - Retained `43,131` valid training transitions across the `300` episodes (`337` batches per epoch at `batch_size = 128`).
2. **Removed All Time-Index Overrides & Open-Loop Hacks ([`training/dataset.py`](./training/dataset.py), [`training/model.py`](./training/model.py), [`evaluation/evaluator.py`](./evaluation/evaluator.py))**:
   - Removed the `step < 16` gripper `0.68` override in both dataset preparation and closed-loop evaluation.
   - Removed `home_anchor_prob`, `pan_noise_std`, and `dq[:, 0] = 0` masking.
   - Configured causal multi-scale proprioception history `proprio_history_lags = (0, 4, 8, 16, 32)` (`0.0 s, 0.2 s, 0.4 s, 0.8 s, 1.6 s`), producing a `30D` history vector (`6D` current posture + `4 x 6D` multi-scale displacements).
3. **Evaluation Horizon, Validation Checkpoint Selection & First-Pass Success ([`evaluation/evaluator.py`](./evaluation/evaluator.py), [`training/trainer.py`](./training/trainer.py))**:
   - Increased default rollout horizon from `154` steps to `max_steps = 280` (`14.0 s` at `20 Hz`, giving `136` steps of slack over the `144`-step expert trajectory).
   - Recorded episode success the first time `env.check_constraints(task_id)["all_constraints_passed"]` holds during closed-loop execution.
   - Selected `best_policy.pt` using deterministic ODE action-chunk MSE (`val_ode_mse`) evaluated in `policy.eval()` mode across `512` validation samples.

### 3.2 Training Summary (`20` Epochs, `sim_only`)

| Metric | Epoch 01 | Epoch 05 | Epoch 10 | Epoch 15 | Epoch 20 (Best) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flow Matching Loss (`loss`)** | `0.71162` | `0.14677` | `0.09883` | `0.08083` | **`0.07355`** |
| **Arm Velocity MSE (`arm`)** | `0.78116` | `0.16055` | `0.10793` | `0.08907` | **`0.08168`** |
| **Gripper Velocity MSE (`grip`)** | `0.57255` | `0.11921` | `0.08062` | `0.06436` | **`0.05729`** |
| **Eval-Mode ODE Chunk MSE (`val_ode_mse`)** | `0.02878` | `0.00440` | `0.00240` | `0.00167` | **`0.00140`** |

### 3.3 Benchmark Evaluation (`120` Episodes Total: `60` Train Split + `60` Held-Out Test Split)

| Split | Task | Strict Pass (`all_constraints_passed`) | Reached Source (`< 2.5 cm`) | Lifted Source (`> 2.0 cm`) | Mean Min Source XY (`cm`) | Mean Max Lift (`cm`) | Mean Final Target XY (`cm`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split (`60` eps)** | Task 0 (`pick_pen_holder_to_bowl`) | `0 / 20 (0.0%)` | `8 / 20` | `0 / 20` | `5.48 cm` | `+0.04 cm` | `19.51 cm` |
| **Train Split (`60` eps)** | Task 1 (`pick_cup_to_bowl`) | `0 / 20 (0.0%)` | `8 / 20` | `1 / 20` | `5.64 cm` | `+0.64 cm` | `15.30 cm` |
| **Train Split (`60` eps)** | Task 2 (`stack_cup_on_cube`) | `0 / 20 (0.0%)` | `8 / 20` | `0 / 20` | `5.07 cm` | `+0.00 cm` | `14.03 cm` |
| **Train Split Total** | **All 3 Tasks (`60` seeds)** | **`0 / 60 (0.0%)`** | **`24 / 60 (40.0%)`** | **`1 / 60 (1.7%)`** | **`5.40 cm`** | **`+0.23 cm`** | **`16.28 cm`** |
| **Test Split (`60` eps)** | Task 0 (`pick_pen_holder_to_bowl`) | **`1 / 20 (5.0%)`** (`test_t0_s9013`) | `5 / 20` | `1 / 20` | `6.24 cm` | `+0.61 cm` | `19.91 cm` |
| **Test Split (`60` eps)** | Task 1 (`pick_cup_to_bowl`) | `0 / 20 (0.0%)` | `6 / 20` | `0 / 20` | `5.68 cm` | `+0.13 cm` | `32.25 cm` |
| **Test Split (`60` eps)** | Task 2 (`stack_cup_on_cube`) | `0 / 20 (0.0%)` | `4 / 20` | `1 / 20` | `6.11 cm` | `+0.57 cm` | `18.26 cm` |
| **Test Split Total** | **All 3 Tasks (`60` seeds)** | **`1 / 60 (1.7%)`** | **`15 / 60 (25.0%)`** | **`2 / 60 (3.3%)`** | **`6.01 cm`** | **`+0.44 cm`** | **`23.47 cm`** |

### 3.4 Key Takeaways from Experiment 01

1. **Second Half of Trajectory (Carry, Release, and Retract) Works End-to-End**:
   - On `test_t0_s9013` (`Task 0`, held-out seed `9013`), where the initial reach landed within `0.62 cm` of the pen holder, the policy grasped the pen holder (`0.09 CLAMPED`), lifted it to `13.5 cm`, carried it into the bowl (`Src->Tgt XY = 1.65 cm`), opened the gripper to `0.74 (OPEN)`, retracted cleanly (`Pinch->Src XY = 5.2 cm`), and passed strict `all_constraints_passed` at `Step 149/280`.
   - Even on failed grasp rollouts like `demo_0101` (`min_src_xy = 1.13 cm`), after clamping empty air beside the cup at `Step 056`, the policy still lifted, carried to the bowl (`Pan q0 = +0.19 rad` vs `GT = +0.21 rad`), opened the gripper to `0.72 (OPEN)`, and retracted.
2. **Primary Bottleneck Identified (The Copycat / Proprioceptive Shortcut Problem)**:
   - Counterfactual testing on `exp01_clean_lags32` between a far-left Task 0 demo (`demo_0001`, `GT_left = -0.88 rad`) and a far-right Task 0 demo (`demo_0005`, `GT_right = +0.74 rad`) proved that the policy is almost completely ignoring the cameras at `Step 00..15` and instead extrapolating `proprio_history`:

| Timestep | True Target Pan (`GT_left` vs `GT_right`) | Original `(Left Cam, Left Proprio)` | **Camera Swapped** `(Right Cam, Left Proprio)` | **Proprio Swapped** `(Left Cam, Right Proprio)` | Camera Sensitivity (`swap_cam_diff`) | Proprio Sensitivity (`swap_prop_diff`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Step 00`** (at `q_home`, `dq = 0`) | `-0.700` vs `+0.587 rad` | **`+0.300 rad`** (wrong sign) | **`+0.354 rad`** | `+0.300 rad` | **`0.0126 rad` (blind)** | `0.0000 rad` |
| **`Step 05`** (`0.25 s` into reach) | `-0.866` vs `+0.726 rad` | `-0.722 rad` | **`-0.570 rad`** (still goes LEFT) | **`+0.633 rad`** (flips to RIGHT) | **`0.0331 rad`** | **`0.1801 rad` (5.4x larger)** |
| **`Step 15`** (`0.75 s` into reach) | `-0.881` vs `+0.742 rad` | `-0.813 rad` | **`-0.348 rad`** | **`+0.270 rad`** | `0.1074 rad` | **`0.1964 rad`** |

   - Why this happens: in `data/sim_demos.h5`, `99.3%` of steps (`143/144`) have non-zero velocity deltas `dq = q(t) - q(t-k)` and lie on the deterministic ray $q(t) = q_{\text{home}} + s(t)(q_{\text{hover\_src}} - q_{\text{home}})$. The MLP achieves `val_ode_mse = 0.00140` by extrapolating `proprio_history` without training the vision encoders.
3. **Secondary Root Cause in [`SpatialSoftmaxConvNet`](./training/model.py#L76-L135) (Post-`GroupNorm` `SiLU` Dilution & `8x8` Grid Resolution)**:
   - Cross-validated ridge probing (`encoder_linear_probe.py`) from `t = 0` keypoints, tokens, or raw `8x8` maps to the `t = 21` hover target `[pan, shoulder]` gave $R^2 \le 0.00$ across all 3 cameras, with fixed-camera keypoint standard deviation across demos of only `0.024` to `0.030`.
   - Controlled localization probes (`encoder_variant_tests.py` on `240` unseen layouts) confirmed that switching `conv4` to `stride = 1` (`16x16` grid), removing the pre-softmax `SiLU()`, and initializing `SpatialSoftmax2d` `temperature = 0.1` reduces 2D object localization error from `5.87 cm` to `2.64 cm` on the existing data size (`0.76 cm` when scaled to `2,400` labeled layouts).

***

## 4. Experiment 02a: Sharp `16x16` Encoder + Single-Frame Proprioception (`exp02a_sharp_nohist`)

* **Branch**: `exp/02a-sharp-encoder-nohist` (merged to `main`)
* **Checkpoint**: `checkpoints/exp02a_sharp_nohist/best_policy.pt` (Epoch 20, identical to `latest_policy.pt`)
* **Benchmark JSON**: `checkpoints/exp02a_sharp_nohist/eval_benchmark_sim_clean.json`
* **Diagnostic GIFs**: `checkpoints/exp02a_sharp_nohist/gifs/` (`demo_0000_task0.gif`, `demo_0101_task1.gif`, `demo_0200_task2.gif`, `test_t1_s19000.gif`)
* **Dataset**: `data/sim_demos.h5` (`300` expert trajectories, `43,131` steps)

### 4.1 Controlled Changes vs. Experiment 01 (`exp01_clean_lags32`)

Only **two** changes were made relative to `exp01` so that any shift in vision reliance and closed-loop benchmark performance can be cleanly attributed:
1. **Sharp `16x16` Vision Encoder ([`training/model.py`](./training/model.py#L34-L135))**:
   - Changed `conv4` from `stride = 2` (`8x8` grid) to `stride = 1` (`16x16` spatial grid).
   - Removed the `SiLU` activation between Task `FiLM` and [`SpatialSoftmax2d`](./training/model.py#L16-L65).
   - Initialized `SpatialSoftmax2d` `temperature = 0.1` and removed the separate `viz_temperature` path so visualization overlays and policy conditioning use the exact same keypoints.
2. **Removed Multi-Step Velocity History ([`training/config.py`](./training/config.py#L37))**:
   - Set `proprio_history_lags = (0,)` (`6D` single-frame proprioception).
   - Kept all other hyperparameters identical to `exp01` (`keypoint_noise = 0.01`, `proprio_drop_prob = 0.10`, `proprio_noise_std = 0.02`, `20` epochs, `max_steps = 280`).

### 4.2 Training Summary (`20` Epochs, `sim_only`)

| Metric | Epoch 01 | Epoch 05 | Epoch 10 | Epoch 15 | Epoch 20 (Best & Last) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flow Matching Loss (`loss`)** | `0.73634` | `0.16191` | `0.10625` | `0.08595` | **`0.07787`** |
| **Arm Velocity MSE (`arm`)** | `0.80008` | `0.17129` | `0.11242` | `0.09159` | **`0.08358`** |
| **Gripper Velocity MSE (`grip`)** | `0.60885` | `0.14313` | `0.09390` | `0.07469` | **`0.06646`** |
| **Eval-Mode ODE Chunk MSE (`val_ode_mse`)** | `0.03982` | `0.00742` | `0.00365` | `0.00252` | **`0.00198`** |

### 4.3 Vision Reliance Probe (`vision_reliance_probe.py`) & Encoder Linear Probe (`encoder_linear_probe.py`)

| Timestep `t` | `exp01` Open-Loop Pan Error (`rad`) | **`exp02a` Open-Loop Pan Error (`rad`)** | `exp01` Camera-Swap Ratio (`1.0 = follows cams`) | **`exp02a` Camera-Swap Ratio (`1.0 = follows cams`)** | `exp01` Freeze-History Shift (`rad`) | **`exp02a` Freeze-History Shift (`rad`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`t = 0`** | `0.345` (`max 0.710`) | **`0.268` (`max 0.607`)** | `0.10` | **`0.34` (`3.4x` higher)** | `0.000` | `0.000` |
| **`t = 10`** | `~0.02` | **`0.023` (`max 0.057`)** | `~0.10` | **`0.35`** | `0.19 to 0.33` | **`0.000`** |
| **`t = 21`** | `~0.02` | **`0.015` (`max 0.032`)** | `~0.15` | **`0.39`** | `0.19 to 0.33` | **`0.000`** |
| **`t = 30`** | `~0.02` | **`0.023` (`max 0.052`)** | `~0.15` | **`0.30`** | `0.19 to 0.33` | **`0.000`** |
| **`t = 45`** | `~0.02` | **`0.020` (`max 0.052`)** | `~0.20` | **`0.35`** | `0.19 to 0.33` | **`0.000`** |
| **`t = 60`** | `~0.03` | **`0.037` (`max 0.088`)** | `~0.20` | **`0.45`** | `0.19 to 0.33` | **`0.000`** |
| **`t = 90`** | `~0.03` | **`0.027` (`max 0.075`)** | `~0.25` | **`0.78`** | `0.19 to 0.33` | **`0.000`** |

* **5-Fold Cross-Validated Ridge Probe at `t = 0` (`encoder_linear_probe.py`)**:
  - Learned `SpatialSoftmax2d` temperatures stayed sharp: `third_person_cam = 0.148`, `overhead_cam = 0.145`, `wrist_cam = 0.143` (vs. `0.93 to 0.99` in `exp01`).
  - Fixed-camera keypoint spread across demos increased from `0.024 to 0.030` in `exp01` to `0.0369` (`overhead_cam`), `0.0586` (`third_person_cam`), and `0.0982` (`wrist_cam`).
  - `overhead_cam` `keypoints` and `token` 5-fold CV $R^2$ for predicting the `t = 21` hover target `pan` (`actions[21, 0]`) from `t = 0` rose from **$\le 0.00$ in `exp01` to $+0.66$ in `exp02a`** (`+0.59` across all `300` clean + DR demos), while radial `shoulder` reach depth (`actions[21, 1]`) $R^2$ remained near `0.0` (`+0.02`).

### 4.4 Side-by-Side Benchmark Evaluation (`exp02a` vs. `exp01`, `120` Episodes Total)

| Split | Task | `exp01` Strict Pass | **`exp02a` Strict Pass** | `exp01` Mean Min Src XY (`cm`) | **`exp02a` Mean Min Src XY (`cm`)** | `exp01` Mean Max Lift (`cm`) | **`exp02a` Mean Max Lift (`cm`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split (`60` eps)** | Task 0 (`pick_pen_holder_to_bowl`) | `0 / 20 (0.0%)` | `0 / 20 (0.0%)` | `5.48 cm` | `6.29 cm` | `+0.04 cm` | `+0.07 cm` |
| **Train Split (`60` eps)** | Task 1 (`pick_cup_to_bowl`) | `0 / 20 (0.0%)` | `0 / 20 (0.0%)` | `5.64 cm` | **`4.83 cm`** | `+0.64 cm` | `+0.10 cm` |
| **Train Split (`60` eps)** | Task 2 (`stack_cup_on_cube`) | `0 / 20 (0.0%)` | `0 / 20 (0.0%)` | `5.07 cm` | **`4.36 cm`** | `+0.00 cm` | **`+1.00 cm`** |
| **Train Split Total** | **All 3 Tasks (`60` seeds)** | `0 / 60 (0.0%)` | `0 / 60 (0.0%)` | `5.40 cm` | **`5.16 cm`** | `+0.23 cm` | **`+0.39 cm`** (`2/60` lifted `> 2 cm` vs `1/60`) |
| **Test Split (`60` eps)** | Task 0 (`pick_pen_holder_to_bowl`) | `1 / 20 (5.0%)` | `0 / 20 (0.0%)` | `6.24 cm` | `6.38 cm` | `+0.61 cm` | `+0.05 cm` |
| **Test Split (`60` eps)** | Task 1 (`pick_cup_to_bowl`) | `0 / 20 (0.0%)` | **`1 / 20 (5.0%)`** (`test_t1_s19000`) | `5.68 cm` | `5.78 cm` | `+0.13 cm` | **`+1.63 cm`** |
| **Test Split (`60` eps)** | Task 2 (`stack_cup_on_cube`) | `0 / 20 (0.0%)` | `0 / 20 (0.0%)` | `6.11 cm` | **`4.26 cm`** | `+0.57 cm` | **`+1.63 cm`** |
| **Test Split Total** | **All 3 Tasks (`60` seeds)** | `1 / 60 (1.7%)` | **`1 / 60 (1.7%)`** | `6.01 cm` | **`5.47 cm`** | `+0.44 cm` | **`+1.10 cm`** (**`6/60` lifted `> 2 cm`** vs `2/60`) |

### 4.5 Key Takeaways from Experiment 02a

1. **Sharp `16x16` Encoder + Single-Frame Proprioception Partially Un-Blinded the Policy**:
   - Step-0 camera-swap ratio rose **`3.4x`** (`0.10 -> 0.34`), and `t = 90` (carry/place) swap ratio reached **`0.78`**.
   - `overhead_cam` `t = 0` linear probe $R^2$ for `pan` jumped from **$\le 0.00$ to $+0.66$**.
   - Held-out test episodes that grasped and lifted the source object (`> 2.0 cm`) **tripled** (`2/60 -> 6/60`), including reaching a far-left cup at `Pan = -0.66 rad` within `0.2 cm` (`test_t1_s19000`) and reaching `demo_0101` within `0.66 cm` (`err = 0.04 rad`).
2. **Remaining Two Bottlenecks to Address in Experiment 02b**:
   - **Sample complexity of multi-object 2D localization**: With only `150` clean + `150` DR layouts across 4 simultaneous objects, `t = 0` `shoulder` reach depth $R^2$ is still near zero (`+0.02`) and `t = 0` camera-swap ratio stops at `0.34` (`pan_err = 0.268 rad`). Pretraining the camera encoders on a few thousand rendered labeled layouts with a training-only position head (`exp02b`) directly targets this gap.
   - **Look-alike states during `lower <-> release` (`249/300` episodes)**: When the jaws clamp around the cup wall, the measured gripper position stays at `0.42 to 0.51` while the command ramps `0.05 -> 0.72` over `12` steps, causing single-frame policies (`lags=(0,)`) to stall or drift after clamping (`demo_0101` and `test_t1_s19000`).


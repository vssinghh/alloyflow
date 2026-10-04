# AlloyFlow: Experiment Log

This log tracks every experiment in **AlloyFlow** on the 6-DoF **SO-ARM101** benchmark (`Task 0: pick_pen_holder_to_bowl`, `Task 1: pick_cup_to_bowl`, `Task 2: stack_cup_on_cube`).

Each experiment follows four sections:
1. **Hypothesis**
2. **Changes**
3. **Results**
4. **What to Try Next**

***

## Master Summary Table

| Run | Goal / Hypothesis | Key Changes | Train Pass (`60` eps) | Test Pass (`60` eps) | Test Lifted `> 2 cm` | Step-0 Camera Sensitivity | Main Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 01 (`exp01_clean_lags32`)** | Clean baseline with complete 8-segment trajectories and 5-frame joint history | Kept all release/retract frames, removed time-index hacks, added `30D` history `lags=(0,4,8,16,32)`, `max_steps=280` | `0 / 60` (`0.0%`) | `1 / 60` (`1.7%`) | `2 / 60` (`3.3%`) | `0.10` (blind; `overhead_cam` $R^2 \le 0.00$) | Carry, release, and retract work end-to-end (`test_t0_s9013`), but the policy copies joint velocity history and ignores the cameras |
| **Exp 02a (`exp02a_sharp_nohist`)** | Un-blind the camera encoder and remove the joint velocity copycat shortcut | `16x16` spatial grid (`conv4` stride 1), removed pre-softmax `SiLU`, `temp=0.1`, single-frame `lags=(0,)` (`6D`) | `0 / 60` (`0.0%`) | `1 / 60` (`1.7%`) | `6 / 60` (`10.0%`) | `0.34` (`3.4x` higher; `overhead_cam` pan $R^2 = +0.66$) | Vision is active and test lifts tripled (`2/60 -> 6/60`), but 300 demos are too few to learn accurate forward reach depth (`shoulder` $R^2 = +0.02$) |
| **Exp 02b (`exp02b_pretrain_loc`)** | Teach the camera encoders exact 2D object positions before and during policy training | Stage 1 pretraining on `3,000` labeled layouts + 4 parallel training-only `12D` position heads co-supervised in Stage 2 (`w=0.5`) | **`9 / 60` (`15.0%`)** | **`5 / 60` (`8.3%`)** | **`30 / 60` (`50.0%`)** | **`0.89` (loc error `0.55 to 0.62 cm`; `overhead_cam` pan $R^2 = +0.96$)** | **Localization is now solved (`107/120` reach `< 2 cm`, `65/120` lift `> 2 cm`, `14/120` strict pass); remaining losses are at grasp-to-lift (`42` eps) and lift-to-pass (`51` eps), i.e. contact and release, not perception** |

***

## Experiment 01: Clean Baseline (`exp01_clean_lags32`)

* **Commit / Checkpoint**: `fc826a0` | `checkpoints/exp01_clean_lags32/best_policy.pt`

### 1. Hypothesis
In early prototype runs, aggressive frame trimming accidentally deleted the frames where the robot opens its gripper and retracts upward at the end of a task. We hypothesized that restoring the full 8-segment trajectory (`approach -> descend -> clamp -> lift -> transit -> lower -> release -> retract`) and giving the policy a 5-frame history of its recent joint positions (`lags = (0, 4, 8, 16, 32)`) would let the robot complete the full pick-and-place cycle without freezing mid-air.

### 2. Changes
* **Dataset ([`training/dataset.py`](./training/dataset.py))**: Kept all grasp, release, and retraction frames (`43,131` steps across `300` demos; `150` clean + `150` domain-randomized) and removed all hardcoded step-index overrides.
* **Model ([`training/config.py`](./training/config.py), [`training/model.py`](./training/model.py))**: Set `proprio_history_lags = (0, 4, 8, 16, 32)` (`30D` input: current `6D` joints + `4` past joint deltas). Fixed `SinusoidalTimeEmbedding` (removed `* 1000` aliasing), added residual `LayerNorm` to `MultiCameraAttention`, and made `obs_proj` a 2-layer MLP.
* **Evaluation ([`evaluation/evaluator.py`](./evaluation/evaluator.py))**: Evaluated `120` closed-loop rollouts (`60` training seeds + `60` unseen test seeds `9000+`) with `max_steps = 280` (`14 s` at `20 Hz`), selecting `best_policy.pt` by eval-mode ODE chunk error (`val_ode_mse = 0.00140`).

### 3. Results

| Split (`60` episodes each) | Strict Pass | Reached Source (`< 2.5 cm`) | Lifted Source (`> 2.0 cm`) | Mean Closest Source XY | Mean Max Lift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split (`20` / task)** | `0 / 60 (0.0%)` | `24 / 60 (40.0%)` | `1 / 60 (1.7%)` | `5.40 cm` | `+0.23 cm` |
| **Held-Out Test Split (`20` / task)** | **`1 / 60 (1.7%)`** (`test_t0_s9013`) | `15 / 60 (25.0%)` | `2 / 60 (3.3%)` | `6.01 cm` | `+0.44 cm` |

* **What worked**: Whenever the initial reach landed on the object (such as held-out test seed `test_t0_s9013`, which reached within `0.62 cm` of the pen holder), the policy grasped it, lifted it `13.5 cm`, carried it into the bowl (`1.65 cm` from center), opened the gripper (`0.74`), and retracted cleanly to pass all constraints.
* **What failed (The Copycat Problem)**: Swapping camera images vs. swapping joint history between a far-left demo (`demo_0001`, pan `-0.88 rad`) and a far-right demo (`demo_0005`, pan `+0.74 rad`) showed that the policy was **ignoring the cameras** and simply extrapolating its own past joint velocity:
  1. **Copycat shortcut from joint history**: On `99.3%` of steps (`143/144`), past joint velocity `q(t) - q(t-k)` already points toward the target. The MLP achieved low training loss (`0.07355`) by copying joint momentum without using vision (`Step 00` camera-swap sensitivity was only `0.0126 rad`).
  2. **Blurred camera keypoints**: In [`SpatialSoftmaxConvNet`](./training/model.py), applying `GroupNorm + SiLU` on a coarse `8x8` grid with `temperature = 1.0` washed out `90%` of the object signal into background cells (`overhead_cam` linear probe $R^2 \le 0.00$ at `Step 00`).

### 4. What to Try Next
1. Remove the velocity history (`proprio_history_lags = (0,)`) so the policy cannot extrapolate past joint motion.
2. Sharpen [`SpatialSoftmaxConvNet`](./training/model.py) by switching `conv4` from `8x8` (`stride=2`) to `16x16` (`stride=1`), removing the pre-softmax `SiLU`, and setting `SpatialSoftmax2d` `temperature = 0.1`.

***

## Experiment 02a: Sharp `16x16` Encoder + No Velocity History (`exp02a_sharp_nohist`)

* **Commit / Checkpoint**: `95b8200` | `checkpoints/exp02a_sharp_nohist/best_policy.pt`

### 1. Hypothesis
If we remove multi-step joint velocity history (`lags = (0,)`) and sharpen the camera keypoint extractor (`16x16` spatial grid, no pre-softmax `SiLU`, `temperature = 0.1`), the policy will no longer be able to coast on joint momentum and will be forced to read the 3 cameras from `Step 00`.

### 2. Changes
* **Vision Encoder ([`training/model.py`](./training/model.py))**: Changed `conv4` to `stride = 1` (`16x16` grid), removed `SiLU` before `SpatialSoftmax2d`, set `init_temperature = 0.1`, and removed the separate `viz_temperature` override so the policy and GIF visualizer use the exact same keypoints.
* **Proprioception ([`training/config.py`](./training/config.py))**: Set `proprio_history_lags = (0,)` (single-frame `6D` joint positions only). All other settings remained identical to Experiment 01.

### 3. Results

| Metric | Experiment 01 (`8x8` + `30D` History) | **Experiment 02a (`16x16` + `6D` Single-Frame)** |
| :--- | :---: | :---: |
| **Step-0 Camera-Swap Ratio** (`1.0` = follows camera) | `0.10` | **`0.34` (`3.4x` higher)** |
| **Step-90 (Carry/Place) Camera-Swap Ratio** | `0.25` | **`0.78` (`3.1x` higher)** |
| **`overhead_cam` Step-0 Linear Probe $R^2$ (`pan`)** | `<= 0.00` | **`+0.66`** (`+0.59` across all 300 demos) |
| **`overhead_cam` Step-0 Linear Probe $R^2$ (`shoulder` depth)** | `<= 0.00` | `+0.02` |
| **Train Split (`60` eps)**: Reached `< 2.5 cm` / Lifted `> 2 cm` / Pass | `24` / `1` / `0` | `22` / **`2`** / `0` |
| **Test Split (`60` eps)**: Reached `< 2.5 cm` / Lifted `> 2 cm` / Pass | `15` / `2` / `1` | **`16` / `6` (`3x` more lifts) / `1`** (`test_t1_s19000`) |
| **Test Split Mean Closest Source XY / Mean Max Lift** | `6.01 cm` / `+0.44 cm` | **`5.47 cm` / `+1.10 cm`** |

* **What worked**:
  - Camera reliance jumped **`3.4x` at `Step 00`** (`0.10 -> 0.34`) and reached **`0.78` during transport (`Step 90`)**.
  - The `overhead_cam` features now predict the horizontal target angle (`pan`) at `Step 00` with **$R^2 = +0.66$** (up from $\le 0.00$ in Exp 01).
  - Held-out test episodes that successfully grasped and lifted the target object **tripled from `2/60` to `6/60`**, including reaching a far-left cup at `Pan = -0.66 rad` within `0.2 cm` (`test_t1_s19000`).
* **What still failed**:
  1. **Poor forward reach depth (`shoulder` $R^2 = +0.02$)**: With only `300` demos (`150` clean + `150` DR) and 4 objects moving simultaneously on the table, end-to-end behavior cloning does not have enough distinct tabletop layouts to teach the CNN exact 2D `(X, Y)` coordinates for all 4 objects (especially radial distance from the robot base).
  2. **Post-grasp stall on single-frame proprioception**: When the gripper clamps around a cup wall, the joint angle stays near `0.42 to 0.51` for `12` steps while grip force builds, causing a memoryless `lags=(0,)` policy to occasionally hesitate after grasping.

### 4. What to Try Next
1. **Experiment 02b (Visual Localization Pretraining + Auxiliary Position Supervision)**: Pretrain the camera encoders on `3,000` rendered tabletop layouts with ground-truth 2D object coordinates and keep an auxiliary position loss active during Flow Matching training so `pan` and `shoulder` reach depth both achieve high $R^2$ from `Step 00`.

***

## Experiment 02b: Stage 1 Vision Pretraining + Co-Supervised Position Loss (`exp02b_pretrain_loc`)

* **Commit / Checkpoint**: `1a4b8bf` | `checkpoints/exp02b_pretrain_loc/best_policy.pt`

### 1. Hypothesis
In Experiment 02a, `300` demonstrations were not enough distinct tabletop scenes for end-to-end Flow Matching alone to teach the CNNs radial depth (`shoulder` $R^2 = +0.02$). If we first pretrain the 3 camera CNNs, task embedding, and camera attention on `3,000` labeled tabletop layouts (`Stage 1`) to predict the `12D` tabletop `(X, Y)` positions of `[source, target, pen_holder, cup, bowl, rubiks_cube]`, and then co-supervise 4 parallel training-only position heads alongside the Flow Matching loss during policy training (`Stage 2`), the vision backbone will lock onto sub-centimeter 2D object coordinates without adding any extra inputs to `z_fused` (`192D`) or any overhead at `20 Hz` inference (`Stage 3`).

### 2. Changes
* **Stage 1 Vision Pretraining ([`training/pretrain_vision.py`](./training/pretrain_vision.py))**: Generated `3,000` labeled 3-camera tabletop scenes (`data/loc_layouts_3000.npz`, `50%` clean + `50%` domain-randomized, `50%` home pose + `50%` sampled demo arm poses) and pretrained `vision_encoders`, `task_emb`, `camera_attention`, and 4 parallel `12D` position heads for `3,000` steps (`batch_size = 64`, `lr = 1e-3`).
* **Training-Only Parallel Side-Branch Heads ([`training/model.py`](./training/model.py))**: Added 3 per-camera position heads (`aux_cam_pos_heads`, each taking `[tok_c(32D), z_task(32D)] -> 12D`) and 1 fused position head (`aux_fused_pos_head`, taking `[attended_cams(96D), z_task(32D)] -> 12D`). Kept `z_fused` strictly `192D` (`[attended_cams(96D), z_prop(64D), z_task(32D)]`) so coordinate predictions never enter the policy trunk and all 4 heads are ignored during inference.
* **Stage 2 Co-Supervision & Replay ([`training/dataset.py`](./training/dataset.py), [`training/flow_matching.py`](./training/flow_matching.py), [`training/trainer.py`](./training/trainer.py))**: Added deterministic per-demo `12D` object `(X, Y)` targets (`obj_xy`, with source object coordinates masked out for `step >= 45` once lifted) plus a `32`-sample localization replay batch from `data/loc_layouts_3000.npz` on every training step:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + 0.5 \cdot \mathcal{L}_{\text{pos}}, \qquad \mathcal{L}_{\text{pos}} = 0.5 \cdot \mathcal{L}_{\text{pos, demo}} + 0.5 \cdot \mathcal{L}_{\text{pos, replay}}$$

### 3. Results

| Metric | Exp 01 (`exp01_clean_lags32`) | Exp 02a (`exp02a_sharp_nohist`) | **Exp 02b (`exp02b_pretrain_loc`)** |
| :--- | :---: | :---: | :---: |
| **Post-Stage-2 Source Localization Error** (`tp` / `ov` / `wr` / `fused`) | N/A | N/A | **`0.55 cm` / `0.60 cm` / `2.12 cm` / `0.62 cm`** |
| **Step-0 Camera-Swap Ratio** (`1.0` = follows camera) | `0.10` | `0.34` | **`0.89` (`8.9x` over Exp 01)** |
| **Step-0 Pan Prediction Error** (`rad`) | `0.345 rad` | `0.268 rad` | **`0.054 rad` (`6.4x` lower)** |
| **`overhead_cam` Step-0 5-Fold CV $R^2$ (`pan`, `shoulder`)** | `-0.17, -0.16` | `+0.59, +0.02` | **`+0.96, +0.68`** |
| **`third_person_cam` Step-0 5-Fold CV $R^2$ (`pan`, `shoulder`)** | `-0.11, -0.19` | `-0.03, +0.06` | **`+0.55, +0.42`** |
| **Train Split (`60` eps)**: Reach `< 2 cm` / Lift `> 2 cm` / **Strict Pass** | `16` / `1` / `0` | `15` / `2` / `0` | **`55 (91.7%)` / `35 (58.3%)` / `9 (15.0%)`** |
| **Test Split (`60` eps)**: Reach `< 2 cm` / Lift `> 2 cm` / **Strict Pass** | `11` / `2` / `1` | `12` / `6` / `1` | **`52 (86.7%)` / `30 (50.0%)` / `5 (8.3%)`** |
| **Combined (`120` eps)**: Reach `< 2 cm` / Lift `> 2 cm` / **Strict Pass** | `27 (22.5%)` / `3 (2.5%)` / `1 (0.8%)` | `27 (22.5%)` / `8 (6.7%)` / `1 (0.8%)` | **`107 (89.2%)` / `65 (54.2%)` / `14 (11.7%)`** |
| **Mean Closest Source XY (`Train` / `Test`)** | `5.40 cm` / `6.01 cm` | `5.32 cm` / `5.47 cm` | **`0.76 cm` / `0.82 cm`** |
| **Mean Max Object Lift (`Train` / `Test`)** | `+0.23 cm` / `+0.44 cm` | `+0.42 cm` / `+1.10 cm` | **`+5.98 cm` / `+4.93 cm`** |

* **What worked**:
  - **2D object localization is now solved**: Across all `300` validation scenes after Stage 2 training, source object localization error is **`0.55 cm` on `third_person_cam`**, **`0.60 cm` on `overhead_cam`**, **`2.12 cm` on `wrist_cam`**, and **`0.62 cm` on the fused attention head** ($R^2 = +0.99$ to $+1.00$).
  - **Reach and lift rates surged across all 3 tasks**: In closed-loop evaluation across `120` episodes (`60` train + `60` held-out test), **`107 / 120` (`89.2%`) reached within `2.0 cm` of the source object** (`mean_src_xy = 0.76 cm` train, `0.82 cm` test), **`65 / 120` (`54.2%`) lifted the object above `2.0 cm`**, and **`14 / 120` (`11.7%`) passed all strict task constraints** (`9 / 60` on train: `3/20` Task 0, `3/20` Task 1, `3/20` Task 2; `5 / 60` on held-out test: `3/20` Task 0, `1/20` Task 1, `1/20` Task 2).
* **Takeaway (Where the Remaining Failures Happen)**:
  Tracing the `120`-episode execution funnel reveals that **perception and initial reach are no longer the bottleneck**:
  1. **Total Episodes**: `120`
  2. **Reached within `2.0 cm`**: **`107 / 120` (`89.2%`)** (`13` episodes lost at initial reach)
  3. **Lifted above `2.0 cm`**: **`65 / 120` (`54.2%`)** (**`42` episodes lost at grasp-to-lift**: the gripper reaches the object within `< 2 cm` but slips off the rim, closes slightly early/late, or stalls during clamp)
  4. **Strict Pass**: **`14 / 120` (`11.7%`)** (**`51` episodes lost at lift-to-pass**: the robot lifts and carries the object toward the goal, but misses the receptacle rim, drops slightly off-center on the cube, or nudges a bystander/target object during placement and release)

  In short, **localization is solved, and the remaining `93` post-reach losses (`42` at grasp-to-lift and `51` at lift-to-pass) are contact, grasp closure, and release precision issues, not perception**.

### 4. What to Try Next
1. **Target the two post-reach bottlenecks (`42` grasp-to-lift losses and `51` lift-to-pass losses)**: Improve fine contact timing, wrist-camera guided grasp alignment, and placement/release trajectories while preserving the Stage 1 + Stage 2 localization pipeline.

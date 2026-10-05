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
| **Exp 02b (`exp02b_pretrain_loc`)** | Teach the camera encoders exact 2D object positions before and during policy training | Stage 1 pretraining on `3,000` labeled layouts + 4 parallel training-only `12D` position heads co-supervised in Stage 2 (`w=0.5`) | `9 / 60` (`15.0%`) | `5 / 60` (`8.3%`) | `30 / 60` (`50.0%`) | `0.89` (loc error `0.55 to 0.62 cm`; `overhead_cam` pan $R^2 = +0.96$) | Localization is solved (`107/120` reach `< 2 cm`, `65/120` lift `> 2 cm`, `14/120` strict pass), but `MultiCameraAttention` overfits on `300` demos |
| **Exp 02c (`exp02c_no_attn`)** | Remove `MultiCameraAttention` and `aux_fused_pos_head` so camera tokens enter `z_fused` directly without mixing distortion | Deleted `MultiCameraAttention` & `aux_fused_pos_head` (`-33,696` params); direct `96D` concat `[tok_tp, tok_ov, tok_wr]`; strictly vision-only Stage 1 | `11 / 60` (`18.3%`) | **`16 / 60` (`26.7%`)** | **`40 / 60` (`66.7%`)** | **Loc error `0.56 cm` (`tp`), `0.72 cm` (`ov`, tgt `0.49 cm`), `1.90 cm` (`wr`)** | **Test strict pass more than tripled (`5/60 -> 16/60`) and combined strict pass nearly doubled (`14/120 -> 27/120 = 22.5%`, `75/120 = 62.5%` lift `> 2 cm`)** |
| **Exp 03 (`exp03_demo_v2`)** | Fix `v1` demo V-bottom grasp, `8`-step clamped release rise, and fixed step-42 close clock via `trajectory_version="v2"` | Data-only change (`data/sim_demos_v2.h5`): flat-bottom `close_in_place` grasp, `open_in_place` release (`0` clamped-rise steps), $\mathcal{U}(0.8, 1.2)$ timing jitter (`trim_stationary=False`), reusing `02c` Stage 1 vision | `14 / 60` (`23.3%`) | `7 / 60` (`11.7%`) | `33 / 60` (`55.0%`) | Loc error `0.57 cm` (`tp`), `0.65 cm` (`ov`), `1.99 cm` (`wr`) | Grasp depth (`+1.21 -> +0.86 cm`), `3b` carry jaw (`0.25 -> 0.43`), and target freezes (`4a`: `7 -> 3`) improved; combined strict pass (`21/120` vs `27/120`) and centered close (`45.8%` vs `50.0%`) tied `02c` within single-seed noise |
| **Exp 04 (`exp04_long_train_60ep`)** | Resolve Stage 2 undertraining (`val_ode_mse` still falling at epoch `20`, train/test equally imprecise at `45.0%` vs `46.7%` centered close, jittery `h=0` predictions) | `epochs 20 -> 60` (`240` flow passes/sample) with cosine LR stretched over `60` epochs (`5.0e-4 -> 2.5e-5`); exact `Exp 03` setup (`data/sim_demos_v2.h5`, no attention, reusing `02c` Stage 1 vision), trained on Colab GPU (`Tesla T4`) | **`31 / 60` (`51.7%`)** | **`22 / 60` (`36.7%`)** | **`52 / 60` (`86.7%`)** | **Loc error `0.47 cm` (`tp`), `0.52 cm` (`ov`, tgt `0.39 cm`), `1.77 cm` (`wr`); `val_ode_mse = 0.00220`** | **Combined strict pass rose `2.52x` (`21/120 -> 53/120 = 44.2%`, flips `+42 / -10`, $p \approx 10^{-5}$), centered close jumped `+27.5 pp` (`45.8% -> 73.3%`, mean close `XY` `2.54 -> 1.17 cm`), rim `XY` fell (`2.51 -> 1.69 cm`), open-loop `FK XY` error dropped `-29% to -36%`, and carry/placement (`50/67` failures) is now the main bottleneck** |

### Noise & Process-Metric Caveat (`120`-Episode Benchmark)
* **Single-seed strict-pass noise (`~±10` episodes)**: On `120` closed-loop rollouts, binary strict pass (`all_constraints_passed`) varies by roughly `±10` episodes between single-seed `20`-epoch retrainings because a `2 mm` rim graze flips an entire trajectory from pass to fail.
* **Near-chance episode overlap across `20`-epoch retrains**: Episode-by-episode comparison showed that shared passes between `20`-epoch runs were near random chance (`Exp 02b` vs. `Exp 02c`: `3` shared passes vs. `3.2` expected by chance; `Exp 02c` vs. `Exp 03`: `2` shared passes vs. `4.7` expected by chance). Consequently, the `Exp 02b -> Exp 02c` (`14 -> 27`) and `Exp 02c -> Exp 03` (`27 -> 21`) shifts are not conclusive on strict pass alone, and `Exp 02c -> Exp 03` is a statistical tie.
* **First unambiguously significant gain (`Exp 04`)**: In `Exp 04` (`21/120 -> 53/120`), two-way episode flips vs. `Exp 03` were **`+42` (`Fail -> Pass`) vs. `-10` (`Pass -> Fail`)** (`11` shared passes, McNemar $p \approx 10^{-5}$), and every dense process metric improved together.
* **Primary evaluation signals going forward**: Dense continuous process metrics (**centered grasp close `< 1.5 cm`**, **close `XY` error**, **rim-entry `XY` at `dz < +4.5 cm`**, and **open-loop `FK XY` error**) are the primary decision signals; binary strict pass is secondary.

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

### 4. What to Try Next
1. **Experiment 02c (Remove `MultiCameraAttention` & `aux_fused_pos_head`)**: In `Exp 02b`, `MultiCameraAttention` used a frozen random `ProprioMLP` during Stage 1 and then re-learned joint-conditioned attention weights on only `300` demos in Stage 2, creating a `9/60` (`15.0%`) vs. `5/60` (`8.3%`) train-to-test gap. Test removing `MultiCameraAttention` and `aux_fused_pos_head` entirely so the 3 camera tokens (`[tok_tp, tok_ov, tok_wr] = 96D`) enter `z_fused` directly without attention mixing.

***

## Experiment 02c: Direct Camera Token Concatenation Without Attention (`exp02c_no_attn`)

* **Commit / Checkpoint**: `f728d47` | `checkpoints/exp02c_no_attn/best_policy.pt`

### 1. Hypothesis
In `Exp 02b`, each camera encoder (`SpatialSoftmaxConvNet`) already produced a clean `32D` spatial token with `0.55 to 0.60 cm` localization accuracy, but passing those three tokens through `MultiCameraAttention(query=cat(z_prop, z_task))` before `z_fused` introduced unnecessary query-key-value mixing conditioned on `z_prop` (`64D`). Worse, during Stage 1 pretraining, `ProprioMLP` was frozen at random initialization while `MultiCameraAttention` was trained through `aux_fused_pos_head`, and then in Stage 2 `ProprioMLP` started training from scratch, shifting the attention query distribution across the `300` demonstrations. We hypothesized that **deleting `MultiCameraAttention` and `aux_fused_pos_head` completely** and concatenating `[tok_third_person, tok_overhead, tok_wrist]` (`96D`) directly into `z_fused` (`192D`) would eliminate attention overfitting and improve held-out test generalization.

### 2. Changes
* **Model ([`training/model.py`](./training/model.py), [`training/__init__.py`](./training/__init__.py))**:
  - Deleted `MultiCameraAttention` (`15,616` parameters) and `aux_fused_pos_head` (`18,080` parameters), reducing total trainable parameters by **`33,696` (`-3.1%`)** from `1,089,735` to **`1,056,039`** (`1,026,435` active at inference).
  - Updated [`extract_obs_features`](./training/model.py) to concatenate `[tok_third_person, tok_overhead, tok_wrist]` (`96D`) directly with `z_prop` (`64D`) and `z_task` (`32D`) to form `z_fused` (`192D`), keeping `wrist_cam_drop_prob = 0.05` on slice `[64:96]` during training.
  - Added vision-only [`encode_vision_tokens`](./training/model.py) so Stage 1 pretraining and localization evaluation never touch `proprio` or `proprio_mlp`.
* **Stage 1 & Stage 2 Losses ([`training/pretrain_vision.py`](./training/pretrain_vision.py), [`training/flow_matching.py`](./training/flow_matching.py))**:
  - Stage 1 pretraining optimizes strictly `task_embedding + camera_encoders + aux_cam_pos_heads` with loss `1.0 * MSE(overhead) + 1.0 * MSE(third_person) + 0.5 * MSE(wrist)`.
  - Stage 2 auxiliary position loss averages the 3 per-camera heads (`overhead = 1.0`, `third_person = 1.0`, `wrist = 0.25`, `w_total = 2.25`) across `0.5 * demo_pos_loss + 0.5 * loc_replay_loss`.
  - All training hyperparameters (`seed = 42`, `epochs = 20`, `batch_size = 128`, `lr = 5e-4`, `val_ode_mse` selection) remained identical to `Exp 02b`.

### 3. Results

| Split / Metric | Exp 02b (With Camera Attention) | **Exp 02c (No Camera Attention)** | Change (`Exp 02b -> Exp 02c`) |
| :--- | :---: | :---: | :---: |
| **Trainable / Inference Parameters** | `1,089,735` / `1,042,051` | **`1,056,039` / `1,026,435`** | **`-33,696` (`-3.1%`)** |
| **Stage 1 Loc Error (`ov_src` / `tp_src` / `wr_src`)** | `0.70 cm` / `0.71 cm` / `2.10 cm` | **`0.78 cm` / `0.74 cm` / `2.23 cm`** | Comparable (`all4 = 0.67 cm`, $R^2 = +0.98, +0.99$) |
| **Post-Stage-2 Loc Error (`ov_src` / `tp_src` / `wr_src`)** | `0.60 cm` / `0.55 cm` / `2.12 cm` | **`0.72 cm` / `0.56 cm` / `1.90 cm`** | Wrist improved `-0.22 cm` (`all4 = 0.57 cm`, $R^2 = +0.99, +1.00$) |
| **Validation ODE Action MSE (`val_ode_mse`)** | `0.00222` | **`0.00221`** | `-0.00001` |
| **Train Strict Pass (`60` eps)** | `9 / 60 (15.0%)` | **`11 / 60 (18.3%)`** | **`+3.3%` (`+2` eps)** |
| Train Task 0 (`pick_pen_holder_to_bowl`) | `3 / 20 (15.0%)` | **`4 / 20 (20.0%)`** | `+5.0%` |
| Train Task 1 (`pick_cup_to_bowl`) | `3 / 20 (15.0%)` | **`6 / 20 (30.0%)`** | `+15.0%` |
| Train Task 2 (`stack_cup_on_cube`) | `3 / 20 (15.0%)` | `1 / 20 (5.0%)` | `-10.0%` |
| **Test Strict Pass (`60` eps)** | `5 / 60 (8.3%)` | **`16 / 60 (26.7%)`** | **`+18.4%` (`+11` eps, `3.2x` higher)** |
| Test Task 0 (`pick_pen_holder_to_bowl`) | `3 / 20 (15.0%)` | **`7 / 20 (35.0%)`** | `+20.0%` |
| Test Task 1 (`pick_cup_to_bowl`) | `1 / 20 (5.0%)` | **`7 / 20 (35.0%)`** | `+30.0%` |
| Test Task 2 (`stack_cup_on_cube`) | `1 / 20 (5.0%)` | **`2 / 20 (10.0%)`** | `+5.0%` |
| **Combined Strict Pass (`120` eps)** | `14 / 120 (11.7%)` | **`27 / 120 (22.5%)`** | **`+10.8%` (`+13` eps, `1.93x` higher)** |
| **Combined Reach `< 2.0 cm` (`120` eps)** | `107 / 120 (89.2%)` (`0.79 cm`) | **`105 / 120 (87.5%)`** (`0.85 cm`) | `-1.7%` |
| **Combined Lift `> 2.0 cm` (`120` eps)** | `65 / 120 (54.2%)` (`+5.46 cm`) | **`75 / 120 (62.5%)`** (`+7.24 cm`) | **`+8.3%` (`+10` eps, `+1.78 cm` mean lift)** |

* **What worked**:
  - **Held-out test strict pass rate more than tripled** from `5/60 (8.3%)` to **`16/60 (26.7%)`** (`7/20` on Task 0, `7/20` on Task 1, `2/20` on Task 2), and **combined strict pass rate nearly doubled** from `14/120 (11.7%)` to **`27/120 (22.5%)`**.
  - **Grasp-to-lift conversion improved significantly**: Combined episodes lifting the object `> 2.0 cm` rose from `65/120 (54.2%)` to **`75/120 (62.5%)`** (`40/60 = 66.7%` on held-out test seeds), and mean max lift increased from `+5.46 cm` to **`+7.24 cm`** (`+7.68 cm` on test).
  - **Simpler, cleaner architecture**: Removing `MultiCameraAttention` eliminated the Stage 1 -> Stage 2 query distribution shift and preserved each camera's `32D` token in its own dedicated slot (`[0:32]`, `[32:64]`, `[64:96]`). *(Note: `Exp 02c` changed two things at once by removing attention in Stage 1 pretraining and removing attention in Stage 2 policy training / inference, so this run does not isolate which of the two changes mattered, and the `14 -> 27` strict-pass jump also sits near single-seed retraining variance.)*
* **Execution Funnel Analysis (`120` Episodes)**:
  - **Exp 02b Funnel**: `120` total $\to$ `107 (89.2%)` reach `< 2 cm` (`-13` missed reach) $\to$ `65 (54.2%)` lift `> 2 cm` (`-42` lost at grasp-to-lift) $\to$ `14 (11.7%)` strict pass (`-51` lost at lift-to-pass).
  - **Exp 02c Funnel**: `120` total $\to$ **`105 (87.5%)` reach `< 2 cm`** (`-15` missed reach) $\to$ **`75 (62.5%)` lift `> 2 cm`** (`-30` lost at grasp-to-lift, cutting grasp failures by `29%`) $\to$ **`27 (22.5%)` strict pass** (`-48` lost at lift-to-pass; `28/120` achieved `task_success`, with `1` episode disqualified for nudging the target by `2.0 cm`).
  - Crucially, among the `48` episodes in `Exp 02c` that lifted `> 2 cm` but failed strict pass, **`17` episodes** carried the source object all the way over the target receptacle (`final_src_tgt_xy = 0.67 to 2.29 cm`, with `+9.8 to +13.8 cm` lift), failing only at final release timing or rim contact.

### 4. What to Try Next
1. **Audit `Exp 02c` failures by exact physical failure mode (`93` failed episodes across `120` rollouts)**:
   - Sorting all `93` failures of `Exp 02c` into 8 Mutually Exclusive, Collectively Exhaustive (MECE) telemetry buckets revealed: `1_missed_xy_reach` (`15`), `2a_premature_high_close` (`21`), `2b_caged_grasp_then_slip` (`9`), `3a_mid_air_slip_transit` (`19`), `3b_mid_air_transit_stall` (`11`), `4a_held_at_target_freeze` (`7`), `4b_rim_bounce_tipped_src` (`7`), and `4c_pushed_tipped_target` (`4`).
   - Comparing the `300` `v1` demonstrations against the `27` passes and `93` failures uncovered 3 structural flaws in `v1` demos (`data/sim_demos.h5`):
     1. **1-Frame V-bottom grasp**: `v1` demos touch `grasp_z` for only `1` frame before immediately rising (`dz_at_close_048 = +0.62 ± 0.22 cm`, `min_dz_while_open = +0.56 ± 0.23 cm`).
     2. **Zero variance in close timing**: Every `v1` demo crosses `grip_cmd < 0.48` at exact step `42.0 ± 0.0` (`std = 0`), teaching the policy to close at step `44 to 47` even when `2.52 cm` high (`2a`).
     3. **8-Step clamped upward pull at release**: During `release_and_rise`, `v1` demos pull the arm upward `+1.8 cm` over `8` steps while the jaws are still clamped (`grip_q = 0.51`) around the placed object, creating `1,964` (`d < 0.25`) to `3,848` (`d < 0.35`) aliased lower/release frames and causing `7` target freezes (`4a`).
2. **Experiment 03 (`exp03_demo_v2`)**: Isolate a data-only change (`trajectory_version = "v2"` in `data/sim_demos_v2.h5`) with flat-bottom grasping, open-in-place release, and per-episode $\mathcal{U}(0.8, 1.2)$ segment timing jitter while keeping the `Exp 02c` model architecture and Stage 1 vision checkpoint unchanged.

***

## Experiment 03: Flat-Bottom Grasp, Open-in-Place Release & Timing Jitter (`exp03_demo_v2`)

* **Commit / Checkpoint**: `a935e0a` | `checkpoints/exp03_demo_v2/best_policy.pt`

### 1. Hypothesis
In `Exp 02c`, `78 / 93` failures occurred after reaching within `2.0 cm` of the source object (`30` failed to lift `> 2 cm`, `30` slipped or stalled in transit, and `18` failed at target placement/release). Trace comparison between the `300` `v1` demonstrations (`data/sim_demos.h5`) and the `120` evaluation rollouts showed that `v1` demos: (a) never pause at the bottom of the grasp (`1`-frame V-bottom), (b) close at fixed step `42.0 ± 0.0` in every episode, and (c) rise `+1.8 cm` for `8` steps at the target while the jaws remain clamped around the object. We hypothesized that regenerating the `300` demonstrations on the exact same seeds with **flat-bottom grasping (`close_in_place`)**, **stationary release (`open_in_place`)**, and **per-episode $\mathcal{U}(0.8, 1.2)$ segment timing jitter** (`data/sim_demos_v2.h5`), while keeping the `Exp 02c` architecture and Stage 1 vision weights unchanged, would deepen grasp height (`dz`), eliminate the `0.25` single-wall rim pinch in `3b`, unlock state-dependent close timing, and reduce target release freezes (`4a`).

### 2. Changes
* **Expert Trajectory Generator ([`collection/sim_expert.py`](./collection/sim_expert.py), [`collection/collector.py`](./collection/collector.py))**:
  - Added `trajectory_version = "v2"` ([`SimExpertPlanner._build_segments_v2`](./collection/sim_expert.py)) and generated [`data/sim_demos_v2.h5`](./data/sim_demos_v2.h5) (`300 / 300` constraint pass rate across the exact same `150` `sim_clean` + `150` `sim_dr` scene seeds as `data/sim_demos.h5`):
    - **Flat-Bottom Grasp**: Descends to `grasp_z` with open gripper (`0.75`), dwells open at `grasp_z` for `2 to 5` steps (`100 to 250 ms`), closes in place at `grasp_z` (`0.75 -> 0.08`), and holds closed (`0.08`) at `grasp_z` for `2 to 3` steps before lifting.
    - **Open-in-Place Release**: Lowers to `place_z` (`tgt_z + 0.056` for `bowl` so jaws open cleanly inside the tapered wall without brushing it) with closed gripper (`0.08`), dwells closed at `place_z` for `2 to 4` steps, opens in place at `place_z` (`0.08 -> 0.75`), holds open for `2` steps, and retreats upward to `q_hover_tgt` with open jaws (`0.75`).
    - **Per-Episode Timing Jitter**: Scales motion segment step counts by $s \sim \mathcal{U}(0.8, 1.2)$ per episode and records per-episode `lift_start_step` in HDF5 attributes.
* **Dataset Loader & Stage 1 Reuse ([`training/dataset.py`](./training/dataset.py), [`training/trainer.py`](./training/trainer.py))**:
  - [`RollingWindowDataset`](./training/dataset.py) auto-detects `trajectory_version == "v2"` from HDF5 attributes, sets `trim_stationary = False` to preserve intentional dwell frames (`42,977` total frames), and masks `src_xy` in `obj_xy_mask` using each episode's exact `lift_start_step`.
  - Loaded `checkpoints/exp02c_no_attn/pretrained_vision.pt` directly for Stage 1 and reset RNG seed `42` before training Stage 2 for `20` epochs (`val_ode_mse = 0.00385`).

#### Pre-Training Verification on `data/sim_demos_v2.h5` (`300` Demos)

| Pre-Training Check Metric (`300` demos) | `v1` (`data/sim_demos.h5`) | `v2` (`data/sim_demos_v2.h5`) |
| :--- | :---: | :---: |
| **1. Constraint Pass Rate** | `300 / 300 (100.0%)` | **`300 / 300 (100.0%)`** (`0` failures; identical `300` seeds) |
| **Episode Length (`T`)** | `144.00 ± 0.00` (`[144, 144]`) | **`143.26 ± 6.03`** (med `143.0`, `[p10, p90] = [135.0, 152.0]`, range `[128, 165]`) |
| **2a. Close-Start Step (`step_close_048`)** | `42.00 ± 0.00` (`std = 0.0`) | **`45.22 ± 3.66`** (med `45.0`, `[p10, p90] = [40.0, 50.0]`, range `[36, 54]`) |
| **2b. `dz` at Close (`dz_at_close_048`, cm)** | `+0.62 ± 0.22` (med `+0.69`, `[+0.25, +0.85]`) | **`+0.44 ± 0.14`** (med `+0.52`, `[p10, p90] = [+0.22, +0.56]`) |
| **2c. Min `dz` While Open (`min_dz_while_open`, cm)** | `+0.56 ± 0.23` (med `+0.65`, `[+0.20, +0.81]`) | **`+0.41 ± 0.15`** (med `+0.50`, `[p10, p90] = [+0.18, +0.52]`) |
| **2d. Carry `grip_q` (`carry_mean_gq`)** | `0.48 ± 0.06` (med `0.53`, `[0.39, 0.53]`) | **`0.48 ± 0.07`** (med `0.53`, `[p10, p90] = [0.39, 0.53]`, min `0.387`) |
| **3a. Target Mid-Ramp Steps (`g_cmd in [0.18, 0.48]`)** | `4.00 ± 0.00` steps (`[4, 4]`) | **`1.00 ± 0.00` steps** (`[1, 1]`) |
| **3b. Clamped-While-Rising Steps at Target** | `8.00 ± 0.00` steps (`+1.8 cm` rise while clamped) | **`0.00 ± 0.00` steps** (`max = 0`) |
| **4. Late-Episode (`t >= 80`) Aliased Frames (`|dt| >= 6`)** | `1,964` (`d < 0.25`) / `3,848` (`d < 0.35`) | **`899` (`-54.2%` at `d < 0.25`) / `2,571` (`-33.2%` at `d < 0.35`)** (`0` at `d < 0.15, |dt| >= 12`) |

### 3. Results

#### A. `120`-Episode Closed-Loop Benchmark (`ode_steps = 5`, `exec_horizon = 1`, `temporal_ensemble = True`)

| Split / Task | `Exp 02c` (`v1` demos) | **`Exp 03` (`v2` demos)** | Delta (`03 - 02c`) | Exp 03 Reach `< 2 cm` | Exp 03 Lift `> 2 cm` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Task 0 (`pick_pen_holder_to_bowl`)** | `4 / 20 (20.0%)` | **`6 / 20 (30.0%)`** | `+2 (+10.0%)` | `16 / 20 (80.0%)` | `10 / 20 (50.0%)` |
| **Train Task 1 (`pick_cup_to_bowl`)** | `6 / 20 (30.0%)` | **`7 / 20 (35.0%)`** | `+1 (+5.0%)` | `20 / 20 (100.0%)` | `15 / 20 (75.0%)` |
| **Train Task 2 (`stack_cup_on_cube`)** | `1 / 20 (5.0%)` | **`1 / 20 (5.0%)`** | `0 (0.0%)` | `19 / 20 (95.0%)` | `11 / 20 (55.0%)` |
| **Train Split Subtotal (`60` eps)** | `11 / 60 (18.3%)` | **`14 / 60 (23.3%)`** | **`+3 (+5.0%)`** | **`55 / 60 (91.7%)`** | **`36 / 60 (60.0%)`** |
| **Test Task 0 (`pick_pen_holder_to_bowl`)** | `7 / 20 (35.0%)` | `0 / 20 (0.0%)` | `-7 (-35.0%)` | `14 / 20 (70.0%)` | `6 / 20 (30.0%)` |
| **Test Task 1 (`pick_cup_to_bowl`)** | `7 / 20 (35.0%)` | `4 / 20 (20.0%)` | `-3 (-15.0%)` | `20 / 20 (100.0%)` | `16 / 20 (80.0%)` |
| **Test Task 2 (`stack_cup_on_cube`)** | `2 / 20 (10.0%)` | **`3 / 20 (15.0%)`** | **`+1 (+5.0%)`** | `19 / 20 (95.0%)` | `11 / 20 (55.0%)` |
| **Test Split Subtotal (`60` eps)** | **`16 / 60 (26.7%)`** | `7 / 60 (11.7%)` | `-9 (-15.0%)` | **`53 / 60 (88.3%)`** | `33 / 60 (55.0%)` |
| **Combined Total (`120` eps)** | **`27 / 120 (22.5%)`** | `21 / 120 (17.5%)` | `-6 (-5.0%)` | **`108 / 120 (90.0%)`** *(02c: `105/120`)* | `69 / 120 (57.5%)` *(02c: `75/120`)* |

#### B. MECE Failure Bucket Comparison (`Exp 02c` vs. `Exp 03`)

| Bucket | `Exp 02c` (`n / 120`) | **`Exp 03` (`n / 120`)** | Delta | `Exp 03` Split (`Tr / Te`) | `Exp 03` Tasks (`T0 / T1 / T2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`0_PASSES`** | `27 (22.5%)` | **`21 (17.5%)`** | `-6` | `14 / 7` | `6 / 11 / 4` |
| **`1_missed_xy_reach`** (`min_src_xy >= 2.0 cm`) | `15 (12.5%)` | **`12 (10.0%)`** | **`-3`** | `5 / 7` | `10 / 0 / 2` |
| **`2a_premature_high_close`** (`reach < 2cm`, `lift < 2cm`, closed high/early) | `21 (17.5%)` | **`23 (19.2%)`** | `+2` | `10 / 13` | `10 / 4 / 9` |
| **`2b_caged_grasp_then_slip`** (`reach < 2cm`, `lift < 2cm`, partial lift/slip) | `9 (7.5%)` | **`16 (13.3%)`** | `+7` | `6 / 10` | `4 / 5 / 7` |
| **`3a_mid_air_slip_transit`** (`lift >= 2cm`, dropped at `tgt_xy >= 3.5cm`, `final_gq < 0.25`) | `19 (15.8%)` | **`15 (12.5%)`** | **`-4`** | `12 / 3` | `3 / 5 / 7` |
| **`3b_mid_air_transit_stall`** (`lift >= 2cm`, stalled at `tgt_xy >= 3.5cm`, `final_gq >= 0.25`) | `11 (9.2%)` | **`11 (9.2%)`** | `0` | `4 / 7` | `1 / 7 / 3` |
| **`4a_held_at_target_freeze`** (`tgt_xy < 3.5cm`, never opened `g_q >= 0.25`) | `7 (5.8%)` | **`3 (2.5%)`** | **`-4 (-57%)`** | `3 / 0` | `0 / 1 / 2` |
| **`4b_rim_bounce_tipped_src`** (`tgt_xy < 3.5cm`, released, `src_object` tipped/bounced) | `7 (5.8%)` | **`14 (11.7%)`** | **`+7 (+100%)`** | `5 / 9` | `6 / 5 / 3` |
| **`4c_pushed_tipped_target`** (`tgt_xy < 3.5cm`, displaced/tipped target receptacle) | `4 (3.3%)` | **`5 (4.2%)`** | `+1` | `1 / 4` | `0 / 2 / 3` |

#### C. Trace Comparison Table (`300` v2 Demos vs. `21` Exp 03 Passes vs. Exp 03 Failure Buckets)

*(All entries show `mean ± std (median, [p10, p90])` across episodes in that group.)*

| Group (`n`) | `step_close_048` | `dz_at_close_048` (cm) | `xy_at_close_048` (cm) | `min_dz_while_open` (cm) | `gq_at_lift_20cm` | `carry_mean_gq` | `carry_mean_dz` (cm) | `carry_min_up_z` | `tgt_max_gcmd` | `tgt_mid_ramp_steps` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_V2_DEMOS_300` (`300`)** | `45.2±3.7` (`45.0`, `[40, 50]`) | `+0.44±0.14` (`+0.52`, `[+0.22, +0.56]`) | `0.24±0.08` (`0.27`, `[0.12, 0.32]`) | `+0.41±0.15` (`+0.50`, `[+0.18, +0.52]`) | `0.50±0.07` (`0.55`, `[0.39, 0.55]`) | `0.48±0.07` (`0.53`, `[0.39, 0.53]`) | `+0.68±0.21` (`+0.81`, `[+0.37, +0.86]`) | `0.99±0.01` (`0.99`, `[0.97, 1.00]`) | `0.75±0.00` (`0.75`, `[0.75, 0.75]`) | `1.0±0.0` (`1.0`, `[1, 1]`) |
| **`0_EXP03_PASSES` (`21`)** | `50.3±4.6` (`50.0`, `[43, 55]`) | **`+1.15±0.86` (`+0.91`, `[+0.26, +1.92]`)** | `0.70±0.50` (`0.55`, `[0.27, 1.22]`) | **`+0.98±0.69` (`+0.86`, `[+0.07, +1.88]`)** | **`0.48±0.07` (`0.50`, `[0.39, 0.55]`)** | **`0.46±0.06` (`0.49`, `[0.39, 0.53]`)** | **`+1.36±0.73` (`+1.20`, `[+0.43, +2.08]`)** | `0.94±0.14` (`0.98`, `[0.96, 0.99]`) | `0.71±0.05` (`0.72`, `[0.69, 0.74]`) | **`8.8±10.1` (`5.0`, `[2, 26]`)** |
| *(Ref: `0_EXP02C_PASSES` `27`)* | *`44.5±1.7` (`44.0`)* | *`+1.30±0.69` (`+1.21`)* | *`0.65±0.35` (`0.61`)* | *`+1.24±0.67` (`+1.21`)* | *`0.45±0.09` (`0.47`)* | *`0.42±0.08` (`0.44`)* | *`+1.59±0.72` (`+1.54`)* | *`0.95±0.07` (`0.98`)* | *`0.66±0.05` (`0.67`)* | *`18.4±14.2` (`16.0`)* |
| **`1_missed_xy_reach` (`12`)** | `41.2±12.9` (`46.0`, `[24, 53]`) | `+0.80±0.68` (`+0.95`, `[-0.07, +1.64]`) | `7.91±4.63` (`6.86`, `[4.69, 11.49]`) | `+8.91` (`n=1`) | N/A | N/A | N/A | N/A | N/A | N/A |
| **`2a_premature_high_close` (`23`)** | `49.1±12.5` (`47.0`, `[41, 57]`) | **`+1.82±0.96` (`+1.77`, `[+0.75, +2.80]`)** | **`4.57±3.65` (`3.62`, `[2.37, 6.25]`)** | `+5.55±4.60` (`+3.85`, `[+1.29, +13.40]`) | N/A | N/A | N/A | N/A | N/A | N/A |
| **`2b_caged_grasp_then_slip` (`16`)** | **`56.6±12.5` (`52.5`, `[42, 74]`)** | `+2.50±1.34` (`+2.52`, `[+0.93, +3.58]`) | `2.17±1.12` (`2.05`, `[1.19, 3.24]`) | `+2.87±3.02` (`+2.47`, `[+0.81, +3.43]`) | N/A | N/A | N/A | N/A | N/A | N/A |
| **`3a_mid_air_slip_transit` (`15`)** | `48.2±6.3` (`48.0`, `[42, 53]`) | `+2.11±1.33` (`+2.77`, `[+0.15, +3.38]`) | `2.34±0.87` (`2.15`, `[1.72, 2.65]`) | `+1.96±1.35` (`+2.06`, `[+0.04, +2.97]`) | `0.23±0.10` (`0.23`, `[0.10, 0.36]`) | `0.21±0.08` (`0.20`, `[0.12, 0.32]`) | `+2.62±1.20` (`+2.98`, `[+1.03, +3.82]`) | `0.37±0.59` (`0.61`, `[-0.55, 0.93]`) | N/A | N/A |
| **`3b_mid_air_transit_stall` (`11`)** | `53.6±4.6` (`53.0`, `[49, 60]`) | **`+1.16±0.93` (`+0.79`, `[+0.22, +2.12]`)** | `1.15±1.10` (`0.88`, `[0.50, 1.43]`) | `+2.15±3.24` (`+1.46`, `[+0.21, +2.59]`) | **`0.45±0.06` (`0.47`, `[0.40, 0.50]`)** | **`0.41±0.05` (`0.43`, `[0.34, 0.46]`)** | `+1.89±1.04` (`+2.07`, `[+0.77, +3.42]`) | `0.63±0.42` (`0.76`, `[-0.08, 1.00]`) | N/A | N/A |
| **`4a_held_at_target_freeze` (`3`)** | `43.3±1.3` (`43.0`, `[42, 45]`) | `+0.74±0.24` (`+0.63`, `[+0.55, +0.98]`) | `0.55±0.26` (`0.62`, `[0.28, 0.78]`) | `+0.74±0.24` (`+0.63`, `[+0.53, +0.98]`) | `0.54±0.02` (`0.54`, `[0.53, 0.56]`) | `0.50±0.01` (`0.50`, `[0.49, 0.51]`) | `+1.42±0.22` (`+1.30`, `[+1.25, +1.64]`) | `0.99±0.00` (`0.99`, `[0.99, 0.99]`) | `0.28±0.17` (`0.33`, `[0.11, 0.43]`) | `41.0±47.1` (`16.0`, `[3, 89]`) |
| **`4b_rim_bounce_tipped_src` (`14`)** | `53.0±6.9` (`52.0`, `[44, 61]`) | `+1.67±0.83` (`+1.70`, `[+0.66, +2.86]`) | `1.03±0.76` (`0.86`, `[0.35, 2.08]`) | `+1.58±0.80` (`+1.57`, `[+0.52, +2.70]`) | `0.39±0.11` (`0.37`, `[0.29, 0.53]`) | `0.36±0.10` (`0.36`, `[0.21, 0.49]`) | `+1.94±0.75` (`+2.00`, `[+0.92, +2.91]`) | `0.76±0.27` (`0.91`, `[0.33, 0.99]`) | **`0.71±0.13` (`0.75`, `[0.69, 0.76]`)** | `22.8±30.8` (`11.0`, `[0, 64]`) |
| **`4c_pushed_tipped_target` (`5`)** | `54.0±7.4` (`52.0`, `[47, 63]`) | `+1.70±1.21` (`+1.12`, `[+0.60, +3.19]`) | `1.13±0.41` (`1.25`, `[0.68, 1.56]`) | `+1.59±1.06` (`+1.12`, `[+0.58, +2.89]`) | `0.48±0.05` (`0.47`, `[0.42, 0.53]`) | `0.44±0.05` (`0.43`, `[0.39, 0.49]`) | `+2.07±0.99` (`+1.72`, `[+1.06, +3.24]`) | `0.67±0.25` (`0.68`, `[0.39, 0.95]`) | `0.62±0.19` (`0.72`, `[0.41, 0.76]`) | `7.2±7.8` (`4.5`, `[1, 16]`) |

* **What worked**:
  - **Target freeze (`4a`) dropped by `-57%` (`7 -> 3`, and `0 / 60` on Test)**: Removing the `8`-step clamped rise cut `tgt_mid_ramp_steps` on passing episodes from `18.4` (median `16.0`) in `Exp 02c` to **`8.8` (median `5.0`)** in `Exp 03`, and raised `tgt_max_gcmd` from `0.66` to **`0.71` (median `0.72`)**.
  - **Grasp height (`dz`) and outer-diameter grasp (`carry_mean_gq`) improved**:
    - On passing episodes, median `min_dz_while_open` dropped from `+1.21 cm` to **`+0.86 cm`**, median `dz_at_close_048` dropped from `+1.21 cm` to **`+0.91 cm`**, and median `carry_mean_gq` rose from `0.44` to **`0.49`**.
    - In `2a_premature_high_close`, median `dz_at_close_048` dropped by `-1.02 cm` (`+2.79 cm` in `02c` to **`+1.77 cm`** in `03`).
    - In `3b_mid_air_transit_stall`, the `0.25 ± 0.09` single-wall rim pinch from `Exp 02c` was eliminated: `3b` in `Exp 03` now achieves **`dz_at_close_048 = +1.16 cm` (median `+0.79 cm`)** and **`carry_mean_gq = 0.41 ± 0.05` (median `0.43`)**.
  - **Close timing (`step_close_048`) unlocked from a fixed clock**: Instead of firing at step `44.5 ± 1.7`, `step_close_048` now adapts from `43.3` (`4a`) and `50.3` (`PASSES`) up to `56.6 ± 12.5` (`p90 = 74.0` in `2b`).
* **What shifted and held back total Test pass (`16/60 -> 7/60`, within single-seed noise)**:
  1. **`4b_rim_bounce_tipped_src` rose from `7` to `14` (`9` on Test)**: *(Untested hypothesis: because `open_in_place` opens rapidly with `tgt_mid_ramp_steps = 1.0` in `v2` demos, `temporal_ensemble = True` may blend the open command as the arm enters the target zone, reaching `tgt_max_gcmd = 0.75` while the carried object is still slightly high or tilted (`carry_min_up_z = 0.76` from a `dz = +1.67 cm` grasp), bouncing off the bowl or cube rim. However, this mechanism was never isolated by an ablation, and a `7 -> 14` count shift on `120` episodes is within single-seed retraining noise.)*
  2. **Horizontal (`XY`) offset at close in `2a` (`23` eps) and `2b` (`16` eps)**: Although `2a` now descends much lower in `Z` before closing (`dz_at_close_048 = +1.82 cm`), its horizontal offset when closing (`xy_at_close_048`) is **`4.57 ± 3.65 cm`** (median `3.62 cm`), and `2b` closes at **`xy_at_close_048 = 2.17 ± 1.12 cm`**.

### 4. What to Try Next
1. Address the two remaining bottlenecks exposed by `Exp 03` on [`data/sim_demos_v2.h5`](./data/sim_demos_v2.h5): (a) horizontal `(X, Y)` centering before closing in `2a` / `2b`, and (b) preventing early/tilted release over the target rim in `4b_rim_bounce_tipped_src` (`14` episodes) and stationary dwell stalls in `3b` (`11` episodes).

***

## Diagnostic & Inference Study 03b: Bottleneck #1 Root-Cause & `temporal_ensemble_decay` Sweep (`2026-10-05`)

### 1. Goal & New Evaluator Telemetry (`evaluation/evaluator.py`)
* **Focus**: Bottleneck #1 (`Closing the Fingers Before the Hand Finishes Centering`), which accounts for `39` Stage-2 failures (`23` in `2a_premature_high_close` + `16` in `2b_caged_grasp_then_slip`) and `15` Stage-3 mid-air drops (`3a_mid_air_slip_transit`).
* **Evaluator Additions** ([`evaluation/evaluator.py`](./evaluation/evaluator.py)):
  - Added `grasp_close_step`, `grasp_close_xy_cm`, and `grasp_close_dz_cm` to `EpisodeEvalResult` (recorded at the first step where `action_6d[5] < 0.48`).
  - Added `centered_grasp_close_count`, `centered_grasp_close_rate` (`grasp_close_xy_cm < 1.5 cm`), `mean_grasp_close_xy_cm`, and `mean_grasp_close_dz_cm` to benchmark reports.

### 2. Root-Cause Findings on Bottleneck #1
1. **Baseline `centered_grasp_close_rate` (`grasp_close_xy_cm < 1.5 cm`)**:
   - **`Exp 02c` (`v1` demos)**: `60 / 120 (50.0%)` (`28/60` Train, `32/60` Test; `mean_close_xy = 1.98 cm`, `mean_close_dz = +1.84 cm`).
   - **`Exp 03` (`v2` demos)**: `55 / 120 (45.8%)` (`27/60` Train, `28/60` Test; `mean_close_xy = 2.54 cm`, `mean_close_dz = +1.61 cm`).
2. **Physical Mechanism (`Open Finger Hits Rim at dz = +4.5 cm, Step ~35`)**:
   - The cup and pen holder have a top rim radius of `~2.3 cm`, while each wide-open gripper finger (`gq = 0.64`) sits `~2.5 cm` out from `pinch_site`.
   - In `2a` and `2b`, as the open hand descends through `dz = +4.5 cm` (`step ~35`), it is already `2.7 to 2.8 cm` off-center horizontally (`~2.2 cm` lateral error perpendicular to the reach path). One open finger comes straight down on top of the object rim before the gripper starts closing at `step ~49`, blocking vertical descent (`dz` stalls at `+3.2 to +3.6 cm`) and knocking the object sideways by `1.2 to 1.6 cm`.
   - Grouping all `Exp 03` episodes by horizontal offset when first entering `dz < +4.5 cm` (`rim_xy`):
     - **Clean entry (`rim_xy < 1.5 cm`, `35` eps)**: **`88.6%` (`31/35`)** centered close (`< 1.5 cm`), **`85.7%` (`30/35`)** lifted (`> 2 cm`), **`37.1%` (`13/35`)** Strict Pass.
     - **Rim-grazing (`1.5 <= rim_xy < 2.5 cm`, `38` eps)**: `47.4%` (`18/38`) centered close, `71.1%` (`27/38`) lifted, `13.2%` (`5/38`) Strict Pass.
     - **Wide of rim (`rim_xy >= 2.5 cm`, `40` eps)**: **`15.0%` (`6/40`)** centered close, **`32.5%` (`13/40`)** lifted, **`7.5%` (`3/40`)** Strict Pass.
3. **Why the Hand Arrives `2.1 to 2.8 cm` Off-Center at `dz = +4.5 cm`**:
   - **Open-Loop Vision-to-Joint (`FK XY`) Error**: Although `aux_cam_pos_heads` localize the object `(X, Y)` to `0.60 cm`, those `12D` coordinates are not fed into `z_fused`, and Flow Matching trains only on joint-angle MSE rather than fingertip Cartesian error. Open-loop evaluation on the `150` training demos shows a **`1.70 cm` (`p90 = 2.64 cm`)** `FK XY` error at hover/descent (`Step 15 -> 30` and `Step 20 -> 35`), and **`3.11 cm`** at `Step 0 -> 15` in `Exp 03` (vs. `2.55 cm` in `Exp 02c` due to unconditioned $\mathcal{U}(0.80, 1.20)$ reach speed jitter).
   - **Zero Horizontal Correction in Expert Demos During Descent**: In `_build_segments_v2`, `90%` of horizontal `(X, Y)` travel finishes during `approach_src` (`steps 0..28`, `0.452 cm/step`), while `descend_src` (`steps 28..42`, `0.104 cm/step`) drops straight down in `Z` from an already-centered hover (`xy = 1.0 cm`). The dataset contains zero corrective examples where the hand is at low `Z` (`dz = +3 to +6 cm`) and `2 cm` off-center horizontally.

### 3. Inference Ablation: `temporal_ensemble_decay` Sweep (`0.05`, `0.15`, `0.30`, `0.50`) on `Exp 03` (`120` Episodes)
Tested whether weighting newer action chunks more heavily (`decay > 0.05` in `TemporalEnsembler`, where `w_age = exp(-decay * age)`) improves close-range visual centering at the rim (`checkpoints/exp03_demo_v2/best_policy.pt`, `ode_steps = 5`):

| Metric | `0.05` (Baseline) | `0.15` | `0.30` | `0.50` |
| :--- | :---: | :---: | :---: | :---: |
| **Rim Entry `< 1.5 cm` (`dz < +4.5 cm`)** | **`36 / 120` (`30.0%`)** | `34 / 120` (`28.3%`) | `21 / 120` (`17.5%`) | `10 / 120` (`8.3%`) |
| **Mean / Median Rim `XY` (`cm`)** | **`2.51` / `2.12 cm`** | `2.55` / `2.23 cm` | `3.05` / `2.86 cm` | `3.49` / `3.07 cm` |
| **Centered Grasp Close (`< 1.5 cm`)** | **`55 / 120` (`45.8%`)** | `51 / 120` (`42.5%`) | `33 / 120` (`27.5%`) | `23 / 120` (`19.2%`) |
| *Train / Test Centered Close* | *`27 / 60`, `28 / 60`* | *`25 / 60`, `26 / 60`* | *`11 / 60`, `22 / 60`* | *`14 / 60`, `9 / 60`* |
| **Mean / Median Close `XY` (`cm`)** | `2.54` / `1.76 cm` | **`2.33` / `1.74 cm`** | `2.97` / `2.27 cm` | `4.06` / `3.04 cm` |
| **Mean Close Height `dz` (`cm`)** | `+1.61 cm` | `+1.55 cm` | `+1.55 cm` | `+2.12 cm` |
| **Mean Close Step** | **`54.1`** | `58.2` | `73.1` | `88.2` |
| **Stage 1: Reached (`< 2.0 cm`)** | **`108 / 120` (`90.0%`)** | `107 / 120` (`89.2%`) | `97 / 120` (`80.8%`) | `98 / 120` (`81.7%`) |
| **Stage 2: Lifted (`> 2.0 cm`)** | **`69 / 120` (`57.5%`)** | `58 / 120` (`48.3%`) | `45 / 120` (`37.5%`) | `29 / 120` (`24.2%`) |
| **Stage 3: Near Target (`< 6.0 cm`)** | **`45 / 120` (`37.5%`)** | `35 / 120` (`29.2%`) | `23 / 120` (`19.2%`) | `18 / 120` (`15.0%`) |
| **Stage 4: Goal Only Pass** | **`21 / 120` (`17.5%`)** | `13 / 120` (`10.8%`) | `0 / 120` (`0.0%`) | `3 / 120` (`2.5%`) |
| **STRICT PASS (All Constraints)** | **`21 / 120` (`17.5%`)** | `13 / 120` (`10.8%`) | `0 / 120` (`0.0%`) | `3 / 120` (`2.5%`) |
| *Train / Test Strict Pass* | *`14 / 60`, `7 / 60`* | *`8 / 60`, `5 / 60`* | *`0 / 60`, `0 / 60`* | *`2 / 60`, `1 / 60`* |
| *Task 0 / Task 1 / Task 2 Pass* | *`6` / `11` / `4`* | *`8` / `4` / `1`* | *`0` / `0` / `0`* | *`0` / `3` / `0`* |
| `2a_premature_high_close` | **`23` (`19.2%`)** | `25` (`20.8%`) | `36` (`30.0%`) | `50` (`41.7%`) |
| `4a_held_at_target_freeze` | **`3` (`2.5%`)** | `10` (`8.3%`) | `13` (`10.8%`) | `12` (`10.0%`) |

* **Takeaway (Why `decay = 0.05` Is Optimal)**:
  1. **Lookahead Overcomes Servo Tracking Lag**: Step `0` (`age = 0`) of each predicted chunk is anchored right next to the current lagging joint state `q_curr`. Putting heavy weight on `age = 0..2` (`decay >= 0.30`) commands targets barely ahead of the lagging servos (`step 35` stalls at `dz = +8.35 to +10.41 cm` instead of `+5.18 cm`, `Mean Close Step` stretches from `54.1` to `88.2`, and `4a_held_at_target_freeze` quadruples from `3` to `12`).
  2. **Variance Reduction Over Flow ODE Noise**: Blending across all `16` overlapping chunks (`decay = 0.05`) averages out `16` independent Gaussian initial draws $x_0 \sim \mathcal{N}(0, I)$, whereas high decay (`0.30`, `0.50`) exposes single-chunk jitter and doubles `2a_premature_high_close` (`23 -> 50`).

***

## Experiment 04: `3x` Longer Training Schedule (`exp04_long_train_60ep`)

* **Commit / Checkpoint**: `9778d5a` | `checkpoints/exp04_long_train_60ep/best_policy.pt`

### 1. Hypothesis
Three diagnostic signals in `Exp 03` (`20` epochs on [`data/sim_demos_v2.h5`](./data/sim_demos_v2.h5)) pointed to Stage 2 undertraining rather than an architectural or objective flaw:
1. **Unconverged validation ODE error**: `Exp 03`'s `val_ode_mse` was still falling steeply at epoch `20` (`0.00441` at Ep 15 $\to$ `0.00385` at Ep 20).
2. **Equal imprecision on training and test seeds**: Centered grasp close (`grasp_close_xy_cm < 1.5 cm`) was `27 / 60 (45.0%)` on the exact training seeds and `28 / 60 (46.7%)` on held-out test seeds, showing that the policy had not yet fit the training demonstrations accurately.
3. **Noisy short-horizon (`h = 0..1`) predictions**: The `temporal_ensemble_decay` sweep (`Study 03b`) and open-loop lead analysis showed that `Exp 03`'s immediate next-step predictions (`h = 0`) suffered from large sideways jitter (`norm_ratio = 1.508` vs. forward `proj_ratio = 0.906` along the demo trajectory).

We hypothesized that training the exact `Exp 03` configuration **`3x` longer (`epochs = 20 -> 60`)** with the cosine learning rate schedule stretched over all `60` epochs would allow `val_ode_mse` to converge, cut open-loop `FK XY` and sideways jitter, and raise centered grasp close (`< 1.5 cm`) by at least `+10` percentage points while lowering rim-entry `XY` error from `2.51 cm`.

### 2. Changes
* **Training Schedule ([`training/config.py`](./training/config.py))**:
  - Changed `epochs: int = 20 -> 60` (`80 -> 240` effective stratified flow passes per transition with `K = 4`).
  - `CosineAnnealingLR` (`T_max = config.epochs`) automatically stretched the cosine decay from `lr = 5.00e-4` to `eta_min = 2.50e-5` across all `60` epochs.
* **Everything Else Unchanged vs. `Exp 03`**:
  - Same dataset ([`data/sim_demos_v2.h5`](./data/sim_demos_v2.h5), `300` `v2` demos, `42,977` transitions, `trim_stationary = False`).
  - Same architecture (no camera attention, direct `96D` camera token concatenation `[tok_tp, tok_ov, tok_wr]`, `3` parallel training-only per-camera `12D` position heads), same losses and weights (`L_CFM + 0.5 * L_pos`, `gripper_weight = 2.5`), same `seed = 42`, and same `val_ode_mse` (`512` samples, `10` Euler steps) checkpoint selection rule.
  - Reused the exact Stage 1 checkpoint (`checkpoints/exp02c_no_attn/pretrained_vision.pt`).
  - **Compute Target Note**: `Exp 04` was trained on a remote Google Colab GPU (`Tesla T4`, CUDA) via [`scripts/train.sh`](./scripts/train.sh) (`~44` minutes for `60` epochs), whereas `Exp 03` was trained locally on Mac MPS.

### 3. Results

#### 1. Centered-Close Rate (`< 1.5 cm`) & Grasp-Close `XY` / `dz` / Step (`120` Episodes)

| Split / Task | `Exp 03` (`20` Ep) Centered Close `< 1.5 cm` | **`Exp 04` (`60` Ep) Centered Close `< 1.5 cm`** | Delta | `Exp 03` Mean / Med Close `XY` (`cm`) | **`Exp 04` Mean / Med Close `XY` (`cm`)** | `Exp 03 -> Exp 04` Mean Close `dz` (`cm`) | `Exp 03 -> Exp 04` Mean Close Step |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined (`120` eps)** | `55 / 120 (45.8%)` | **`88 / 120 (73.3%)`** | **`+33 (+27.5 pp)`** | `2.54 / 1.76 cm` | **`1.17 / 0.81 cm`** | `+1.61 -> +1.37 cm` | `54.1 -> 46.3` |
| **Train Split (`60` eps)** | `27 / 60 (45.0%)` | **`46 / 60 (76.7%)`** | **`+19 (+31.7 pp)`** | `2.36 / 1.78 cm` | **`1.06 / 0.77 cm`** | `+1.57 -> +1.35 cm` | `51.6 -> 45.7` |
| **Test Split (`60` eps)** | `28 / 60 (46.7%)` | **`42 / 60 (70.0%)`** | **`+14 (+23.3 pp)`** | `2.72 / 1.66 cm` | **`1.28 / 0.89 cm`** | `+1.64 -> +1.40 cm` | `56.6 -> 46.9` |
| **`Task 0: pick_pen_holder_to_bowl` (`40`)** | `14 / 40 (35.0%)` (`Tr 8/20, Te 6/20`) | **`30 / 40 (75.0%)`** (`Tr 14/20, Te 16/20`) | **`+16 (+40.0 pp)`** | `4.05 / 3.11 cm` | **`1.30 / 0.94 cm`** | `+1.41 -> +1.39 cm` | `47.5 -> 45.2` |
| **`Task 1: pick_cup_to_bowl` (`40`)** | `26 / 40 (65.0%)` (`Tr 12/20, Te 14/20`) | `25 / 40 (62.5%)` (`Tr 14/20, Te 11/20`) | `-1 (-2.5 pp)` | `1.50 / 1.00 cm` | **`1.26 / 0.97 cm`** | `+1.35 -> +1.15 cm` | `56.5 -> 46.9` |
| **`Task 2: stack_cup_on_cube` (`40`)** | `15 / 40 (37.5%)` (`Tr 7/20, Te 8/20`) | **`33 / 40 (82.5%)`** (`Tr 18/20, Te 15/20`) | **`+18 (+45.0 pp)`** | `2.12 / 2.13 cm` | **`0.96 / 0.76 cm`** | `+2.06 -> +1.58 cm` | `58.1 -> 46.8` |

#### 2. Off-Center Error at Rim Entry (`dz < +4.5 cm`)

| Split | `Exp 03` (`20` Ep) Entered Rim | **`Exp 04` (`60` Ep) Entered Rim** | `Exp 03` Rim Entry `< 1.5 cm` | **`Exp 04` Rim Entry `< 1.5 cm`** | Delta | `Exp 03` Mean / Med Rim `XY` (`cm`) | **`Exp 04` Mean / Med Rim `XY` (`cm`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined (`120` eps)** | `116 / 120` | **`119 / 120`** | `36 / 120 (30.0%)` | **`55 / 120 (45.8%)`** | **`+19 (+15.8 pp)`** | `2.51 / 2.12 cm` (`p90 = 4.52`) | **`1.69 / 1.55 cm` (`p90 = 2.86`)** |
| **Train Split (`60` eps)** | `57 / 60` | **`60 / 60`** | `15 / 60 (25.0%)` | **`27 / 60 (45.0%)`** | **`+12 (+20.0 pp)`** | `2.46 / 2.11 cm` | **`1.58 / 1.54 cm`** |
| **Test Split (`60` eps)** | `59 / 60` | **`59 / 60`** | `21 / 60 (35.0%)` | **`28 / 60 (46.7%)`** | **`+7 (+11.7 pp)`** | `2.56 / 2.13 cm` | **`1.80 / 1.56 cm`** |

#### 3. Open-Loop Policy Lead vs. Demo Lead (`150` `sim_clean` Demos, `h = 0, 1, 2, 3`)

For each horizon step $h \in \{0, 1, 2, 3\}$ on demo frames (`steps 0..50` approach/descent and `overall` across all active arm steps where $\|\mathbf{a}^{\text{demo}}_{t+h} - \mathbf{q}_t\|_2 > 0.01\text{ rad}$), we compare:
* **`norm_ratio`**: $\|\hat{\mathbf{a}}_{t+h} - \mathbf{q}_t\|_2 / \|\mathbf{a}^{\text{demo}}_{t+h} - \mathbf{q}_t\|_2$ (total step magnitude relative to the demo lead, including any sideways jitter).
* **`proj_ratio`**: $\frac{(\hat{\mathbf{a}}_{t+h} - \mathbf{q}_t) \cdot (\mathbf{a}^{\text{demo}}_{t+h} - \mathbf{q}_t)}{\|\mathbf{a}^{\text{demo}}_{t+h} - \mathbf{q}_t\|_2^2}$ (signed forward progress strictly along the true demo trajectory direction).

| Phase | Horizon `h` | `Exp 03` Pred / Demo Norm (`rad`) | **`Exp 04` Pred / Demo Norm (`rad`)** | `Exp 03` `norm_ratio` (Mean / Med) | **`Exp 04` `norm_ratio` (Mean / Med)** | `Exp 03` Forward `proj_ratio` | **`Exp 04` Forward `proj_ratio`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approach / Descent (`steps 0..50`)** | `h = 0` | `0.1080 / 0.0716` | **`0.0892 / 0.0716`** | `1.508 / 1.544` | **`1.246 / 1.262`** | `0.906` | **`0.939`** |
| **Approach / Descent (`steps 0..50`)** | `h = 1` | `0.1308 / 0.1075` | **`0.1194 / 0.1075`** | `1.216 / 1.254` | **`1.111 / 1.140`** | `0.919` | **`0.962`** |
| **Approach / Descent (`steps 0..50`)** | `h = 2` | `0.1627 / 0.1438` | **`0.1541 / 0.1438`** | `1.131 / 1.168` | **`1.071 / 1.098`** | `0.978` | **`0.983`** |
| **Approach / Descent (`steps 0..50`)** | `h = 3` | `0.1977 / 0.1797` | **`0.1895 / 0.1797`** | `1.100 / 1.140` | **`1.055 / 1.082`** | `0.991` | **`1.004`** |
| **Overall (All Active Steps)** | `h = 0` | `0.1073 / 0.0680` | **`0.0875 / 0.0680`** | `1.577 / 1.557` | **`1.286 / 1.275`** | `0.986` | **`1.009`** |
| **Overall (All Active Steps)** | `h = 1` | `0.1235 / 0.1006` | **`0.1122 / 0.1006`** | `1.227 / 1.233` | **`1.115 / 1.121`** | `1.015` | **`1.010`** |
| **Overall (All Active Steps)** | `h = 2` | `0.1502 / 0.1333` | **`0.1422 / 0.1333`** | `1.127 / 1.144` | **`1.067 / 1.086`** | `1.013` | **`1.011`** |
| **Overall (All Active Steps)** | `h = 3` | `0.1801 / 0.1657` | **`0.1738 / 0.1657`** | `1.087 / 1.112` | **`1.049 / 1.067`** | `1.013` | **`1.022`** |

* **Lead-Test Finding (`Jittery, Not Timid`)**: `Exp 03`'s `h = 0` predictions were **jittery, not timid**. During approach and descent (`steps 0..50`), `Exp 03` had a raw `L2` step norm ratio of `1.508` (`median = 1.544`) while its signed forward projection along the demo trajectory was only `0.906` (meaning $\sqrt{1.508^2 - 0.906^2} \approx 1.21\times$ of the step norm was orthogonal jitter). Training for `60` epochs in `Exp 04` cut that sideways excess roughly in half (`norm_ratio` `1.508 -> 1.246`, `median` `1.544 -> 1.262`) while simultaneously increasing forward projection (`0.906 -> 0.939` at `h = 0`, `0.919 -> 0.962` at `h = 1`).

#### 4. Open-Loop Forward-Kinematics (`FK`) Fingertip `XY` & `3D XYZ` Error (`150` `sim_clean` Demos)

| Trajectory Phase (`step`, horizon `h`) | `Exp 03` Mean / Med / `p90` `FK XY` (`cm`) | **`Exp 04` Mean / Med / `p90` `FK XY` (`cm`)** | `FK XY` Reduction | `Exp 03 -> Exp 04` Mean `3D XYZ` (`cm`) | `Exp 03 -> Exp 04` Joint `L1` (`rad`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`step0_h15_approach` (`t = 0, h = 15`)** | `1.95 / 1.76 / 3.63 cm` | **`1.40 / 1.26 / 2.45 cm`** | **`-28.5%`** | `2.33 -> 1.62 cm (-30.6%)` | `0.0642 -> 0.0419 rad` |
| **`step15_h15_hover` (`t = 15, h = 15`)** | `1.02 / 0.91 / 1.78 cm` | **`0.65 / 0.62 / 1.13 cm`** | **`-36.3%`** | `1.69 -> 1.34 cm (-20.4%)` | `0.0528 -> 0.0422 rad` |
| **`step20_h15_hover_descend` (`t = 20, h = 15`)** | `0.91 / 0.78 / 1.71 cm` | **`0.59 / 0.52 / 1.04 cm`** | **`-35.9%`** | `1.47 -> 1.03 cm (-29.7%)` | `0.0487 -> 0.0365 rad` |
| **`grasp_bot_h0` (`t = step_close, h = 0`)** | `0.74 / 0.61 / 1.32 cm` | **`0.48 / 0.40 / 0.90 cm`** | **`-35.3%`** | `0.96 -> 0.62 cm (-35.3%)` | `0.0258 -> 0.0177 rad` |

* **Post-Stage-2 Visual Localization (`300` Demo Frame-0 Layouts)**: `overhead_cam` source error dropped from `0.65 cm` (`Exp 03`) to **`0.52 cm`** (`tgt = 0.39 cm`, `all4 = 0.44 cm`, $R^2 = +0.99, +1.00$), `third_person_cam` source error dropped from `0.57 cm` to **`0.47 cm`**, and `wrist_cam` source error dropped from `1.99 cm` to **`1.77 cm`**.

#### 5. `120`-Episode Closed-Loop Execution Funnel (By Split & Task)

*(Note: `Exp 04` had `0` constraint-only failures across all `120` episodes, so `Goal Only Pass = Strict Pass = 53 / 120`.)*

| Split / Task (`n`) | Stage 1: Reach `< 2.0 cm` (`03 -> 04`) | Stage 1b: Centered Close `< 1.5 cm` (`03 -> 04`) | Stage 2: Lift `> +2.0 cm` (`03 -> 04`) | Stage 3: Near Target `< 6.0 cm` (`03 -> 04`) | **Stage 4: Strict Pass (`03 -> 04`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined (`120` eps)** | `108 (90.0%) ->` **`116 (96.7%)`** | `55 (45.8%) ->` **`88 (73.3%)`** | `69 (57.5%) ->` **`103 (85.8%)`** | `45 (37.5%) ->` **`77 (64.2%)`** | `21 (17.5%) ->` **`53 (44.2%, +26.7 pp)`** |
| **Train Split (`60` eps)** | `55 (91.7%) ->` **`58 (96.7%)`** | `27 (45.0%) ->` **`46 (76.7%)`** | `39 (65.0%) ->` **`51 (85.0%)`** | `24 (40.0%) ->` **`42 (70.0%)`** | `14 (23.3%) ->` **`31 (51.7%, +28.3 pp)`** |
| **Test Split (`60` eps)** | `53 (88.3%) ->` **`58 (96.7%)`** | `28 (46.7%) ->` **`42 (70.0%)`** | `30 (50.0%) ->` **`52 (86.7%)`** | `21 (35.0%) ->` **`35 (58.3%)`** | `7 (11.7%) ->` **`22 (36.7%, +25.0 pp)`** |
| **`Task 0: pick_pen_holder_to_bowl` (`40`)** | `30 (75.0%) ->` **`39 (97.5%)`** | `14 (35.0%) ->` **`30 (75.0%)`** | `16 (40.0%) ->` **`32 (80.0%)`** | `11 (27.5%) ->` **`29 (72.5%)`** | `6 (15.0%) ->` **`22 (55.0%)`** (`Tr 10/20, Te 12/20`) |
| **`Task 1: pick_cup_to_bowl` (`40`)** | `40 (100.0%) ->` `38 (95.0%)` | `26 (65.0%) ->` `25 (62.5%)` | `31 (77.5%) ->` **`34 (85.0%)`** | `21 (52.5%) ->` **`23 (57.5%)`** | `11 (27.5%) ->` **`14 (35.0%)`** (`Tr 9/20, Te 5/20`) |
| **`Task 2: stack_cup_on_cube` (`40`)** | `38 (95.0%) ->` **`39 (97.5%)`** | `15 (37.5%) ->` **`33 (82.5%)`** | `22 (55.0%) ->` **`37 (92.5%)`** | `13 (32.5%) ->` **`25 (62.5%)`** | `4 (10.0%) ->` **`17 (42.5%)`** (`Tr 12/20, Te 5/20`) |

#### 6. MECE Failure Buckets & Two-Way Episode Flips (`Exp 03` vs. `Exp 04`)

| Bucket | `Exp 03` (`20` Ep) | **`Exp 04` (`60` Ep)** | Delta (`04 - 03`) | `Exp 04` Split (`Tr / Te`) | `Exp 04` Tasks (`T0 / T1 / T2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`0_PASS`** | `21 (17.5%)` | **`53 (44.2%)`** | **`+32 (+26.7 pp)`** | `31 / 22` | `22 / 14 / 17` |
| **`1_missed_xy_reach`** (`min_src_xy >= 2.0 cm`) | `12 (10.0%)` | **`4 (3.3%)`** | **`-8 (-67%)`** | `2 / 2` | `1 / 2 / 1` |
| **`2a_premature_high_close`** (`reach < 2cm`, `lift < 2cm`, closed high/early) | `23 (19.2%)` | **`5 (4.2%)`** | **`-18 (-78%)`** | `3 / 2` | `2 / 2 / 1` |
| **`2b_caged_grasp_then_slip`** (`reach < 2cm`, `lift < 2cm`, partial lift/slip) | `16 (13.3%)` | **`8 (6.7%)`** | **`-8 (-50%)`** | `4 / 4` | `5 / 2 / 1` |
| **`3a_mid_air_slip_transit`** (`lift >= 2cm`, dropped at `tgt_xy >= 3.5cm`, `final_gq < 0.25`) | `15 (12.5%)` | **`11 (9.2%)`** | **`-4 (-27%)`** | `4 / 7` | `2 / 5 / 4` |
| **`3b_mid_air_transit_stall`** (`lift >= 2cm`, stalled at `tgt_xy >= 3.5cm`, `final_gq >= 0.25`) | `11 (9.2%)` | `13 (10.8%)` | `+2` | `5 / 8` | `1 / 6 / 6` |
| **`4a_held_at_target_freeze`** (`tgt_xy < 3.5cm`, never opened `g_q >= 0.25`) | `3 (2.5%)` | **`0 (0.0%)`** | **`-3 (-100%)`** | `0 / 0` | `0 / 0 / 0` |
| **`4b_rim_bounce_tipped_src`** (`tgt_xy < 3.5cm`, released, `src_object` tipped/bounced) | `14 (11.7%)` | `17 (14.2%)` | `+3` | `7 / 10` | `7 / 6 / 4` |
| **`4c_pushed_tipped_target`** (`tgt_xy < 3.5cm`, displaced/tipped target receptacle) | `5 (4.2%)` | `9 (7.5%)` | `+4` | `4 / 5` | `0 / 3 / 6` |

* **Two-Way Episode Flips (`Exp 03 -> Exp 04` Across All `120` Episodes)**:
  - **`Pass -> Pass` (Retained Passes)**: **`11`** episodes (`7` Train, `4` Test).
  - **`Fail -> Pass` (New Passes Gained)**: **`+42`** episodes (`+24` Train, `+18` Test; `18` on `Task 0: pick_pen_holder_to_bowl`, `10` on `Task 1: pick_cup_to_bowl`, `14` on `Task 2: stack_cup_on_cube`), recovering `16` former `2a/2b` grasp failures, `11` former `3a/3b` transit failures, `11` former `4a/4b/4c` target failures, and `4` former `1_missed_xy_reach` episodes.
  - **`Pass -> Fail` (Passes Lost)**: **`-10`** episodes (`-7` Train, `-3` Test; `2` on `Task 0: pick_pen_holder_to_bowl`, `7` on `Task 1: pick_cup_to_bowl`, `1` on `Task 2: stack_cup_on_cube`).
  - **`Fail -> Fail`**: **`57`** episodes (`26` Train, `31` Test).
  - **Net Gain**: **`+32` passes** (`+42` vs. `-10`, two-sided McNemar $p \approx 10^{-5}$).

#### 7. Training Curve: `val_ode_mse` Convergence Across `60` Epochs

| Epoch Milestone | Learning Rate (`CosineAnnealingLR`) | Train Loss (`total / cfm / pos`) | Validation `val_ode_mse` (`10`-Step ODE) | Best Checkpoint Saved? |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch `01`** | `4.997e-4` | `0.38866` (`0.29951` / `0.17830`) | `0.02581` | Yes (`0.02581`) |
| **Epoch `10`** | `4.681e-4` | `0.12994` (`0.09771` / `0.06446`) | `0.00462` | Yes (`0.00462`) |
| **Epoch `20`** *(Exp 03 endpoint)* | `3.813e-4` | `0.10457` (`0.08378` / `0.04157`) | `0.00368` *(Exp 03 was `0.00385`)* | Yes (`0.00368`) |
| **Epoch `30`** | `2.625e-4` | `0.09285` (`0.07587` / `0.03396`) | `0.00315` | Yes (`0.00308` at Ep 28) |
| **Epoch `40`** | `1.438e-4` | `0.08190` (`0.06794` / `0.02793`) | `0.00249` | Yes (`0.00249`) |
| **Epoch `50`** | `5.693e-5` | `0.07721` (`0.06459` / `0.02523`) | `0.00237` | Yes (`0.00237`) |
| **Epoch `54`** | `3.655e-5` | `0.07625` (`0.06394` / `0.02463`) | `0.00222` | Yes (`0.00222`) |
| **Epoch `60` (Final)** | `2.500e-5` | `0.07510` (`0.06308` / `0.02404`) | **`0.00220`** | **Yes (`0.00220`, `-42.9%` vs. Exp 03)** |

* **Did `val_ode_mse` level off? Yes**: Between epochs `01` and `45`, `val_ode_mse` fell rapidly (`0.02581 -> 0.00368` at Ep 20 $\to$ `0.00242` at Ep 45), then plateaued cleanly across epochs `54` to `60` in a tight band (`0.00220` to `0.00235`) as the cosine learning rate reached `2.50e-5`.
* **Decision Rule Applied**: Centered grasp close rose by **`+27.5` percentage points** (`45.8% -> 73.3%`, well above the `+10` point threshold) and mean rim `XY` fell from `2.51 cm` to **`1.69 cm`**. We are in **Case 1 (Accept)**: `checkpoints/exp04_long_train_60ep/best_policy.pt` (`60` epochs) is promoted to the new `Sim-Only` baseline without needing a second training seed.

### 4. What to Try Next
1. **Carry and target placement are now the primary bottleneck (`50 / 67` failures)**: Pre-lift reach and grasp failures (`1_missed_xy_reach + 2a_premature_high_close + 2b_caged_grasp_then_slip`) dropped by **`-66.7%`** (`51 -> 17` episodes, or `42.5% -> 14.2%` of all rollouts), and `4a_held_at_target_freeze` dropped to **`0`**. Now **`50` of the `67` remaining failures (`74.6%`)** happen after lift during transport or placement (`3a_mid_air_slip_transit = 11`, `3b_mid_air_transit_stall = 13`, `4b_rim_bounce_tipped_src = 17`, `4c_pushed_tipped_target = 9`), because `85.8%` (`103/120`) of episodes now lift the object and reach the transport/placement stages.
2. **Address the emerging train-to-test generalization gap (`51.7%` vs. `36.7%`)**: Once underfitting was removed by training for `60` epochs, a `15.0`-point train/test gap appeared on strict pass (`31/60 = 51.7%` train vs. `22/60 = 36.7%` test) and target approach (`42/60 = 70.0%` train vs. `35/60 = 58.3%` test reaching `< 6.0 cm` of target). Collecting more simulation demonstrations per task is a direct candidate to close this gap.
3. **Investigate `Task 1: pick_cup_to_bowl` (`35.0%` strict pass)**: While `Task 0: pick_pen_holder_to_bowl` (`35.0% -> 75.0%` centered close, `55.0%` strict pass) and `Task 2: stack_cup_on_cube` (`37.5% -> 82.5%` centered close, `42.5%` strict pass) surged, `Task 1: pick_cup_to_bowl` centered-close rate did not improve (`26/40 = 65.0% -> 25/40 = 62.5%`) and it is now the weakest task (`14/40 = 35.0%` strict pass, with `11` mid-air transit slips/stalls `3a + 3b`).
4. **Auxiliary FK loss remains on hold**: Because `60`-epoch training alone reduced open-loop `FK XY` error by `-28.5% to -36.3%` (down to `0.48 to 0.65 cm` at hover/grasp) and lifted centered close to `73.3%`, adding a differentiable FK fingertip loss is on hold.


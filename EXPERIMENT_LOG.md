# AlloyFlow: Experiment Log

This log tracks every experiment in **AlloyFlow** on the 6-DoF **SO-ARM101** benchmark (`Task 0: pick_pen_holder_to_bowl`, `Task 1: pick_cup_to_bowl`, `Task 2: stack_cup_on_cube`).

Each experiment is evaluated on **`120` closed-loop episodes** (`60` training seeds + `60` unseen test seeds) and follows four short sections:
1. **Hypothesis**
2. **Changes**
3. **Results**
4. **What to Try Next**

***

## Master Summary Table

| Run | Key Change | Reach `< 2 cm` (`120` eps) | Lift `> 2 cm` (`120` eps) | Train Pass (`60` eps) | Test Pass (`60` eps) | Total Pass (`120` eps) | Main Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 01 (`exp01_clean_lags32`)** | Full 8-segment demos + 5-frame joint history (`30D`) | `27 / 120` (`22.5%`) | `3 / 120` (`2.5%`) | `0 / 60` (`0.0%`) | `1 / 60` (`1.7%`) | `1 / 120` (`0.8%`) | Policy ignores cameras and copies past joint velocity |
| **Exp 02a (`exp02a_sharp_nohist`)** | Sharp `16x16` camera grid (`temp=0.1`) + single-frame joints (`6D`) | `27 / 120` (`22.5%`) | `8 / 120` (`6.7%`) | `0 / 60` (`0.0%`) | `1 / 60` (`1.7%`) | `1 / 120` (`0.8%`) | Un-blinds horizontal angle (`pan`), but `300` demos are too few to learn forward depth |
| **Exp 02b (`exp02b_pretrain_loc`)** | Stage 1 vision pretraining (`3,000` scenes) + Stage 2 auxiliary position loss | `107 / 120` (`89.2%`) | `65 / 120` (`54.2%`) | `9 / 60` (`15.0%`) | `5 / 60` (`8.3%`) | `14 / 120` (`11.7%`) | Solves 2D object localization (`0.6 cm` error), but camera attention overfits on `300` demos |
| **Exp 02c (`exp02c_no_attn`)** | Removed `MultiCameraAttention`; direct `96D` camera token concatenation | `105 / 120` (`87.5%`) | `75 / 120` (`62.5%`) | `11 / 60` (`18.3%`) | `16 / 60` (`26.7%`) | `27 / 120` (`22.5%`) | Simpler model doubles total pass rate, but `v1` demos close on a fixed clock and pull up while clamped |
| **Exp 03 (`exp03_demo_v2`)** | `v2` demos (`data/sim_demos_v2.h5`): flat-bottom grasp, open-in-place release, timing jitter | `108 / 120` (`90.0%`) | `69 / 120` (`57.5%`) | `14 / 60` (`23.3%`) | `7 / 60` (`11.7%`) | `21 / 120` (`17.5%`) | Fixes target release freezes and deepens grasps, but 20 epochs leaves the policy undertrained |
| **Exp 04 (`exp04_long_train_60ep`)** | Trained `3x` longer (`20 -> 60` epochs) on `v2` demos | `116 / 120` (`96.7%`) | `103 / 120` (`85.8%`) | `31 / 60` (`51.7%`) | `22 / 60` (`36.7%`) | `53 / 120` (`44.2%`) | Centered grasps jump `46% -> 73%` and task success more than doubles (`17.5% -> 44.2%`) |
| **Exp 05 (`exp05_wide_jaw`, rejected)** | Wider gripper opening during approach/descent (`0.65 -> 0.85`, `data/sim_demos_v3.h5`) | `96 / 120` (`80.0%`) | `74 / 120` (`61.7%`) | `21 / 60` (`35.0%`) | `21 / 60` (`35.0%`) | `42 / 120` (`35.0%`) | Single-hinge geometry shifts jaw center `14.9 mm` sideways during close, causing side-swiping misses |
| **Exp 06 (`exp06_dart`)** | DART source recovery demos (`data/sim_demos_v2_dart.h5`): $\delta \sim \mathcal{U}(0, 2.5\text{ cm})$ hover offset with clean `v2` labels | `116 / 120` (`96.7%`) | `103 / 120` (`85.8%`) | `36 / 60` (`60.0%`) | `26 / 60` (`43.3%`) | `62 / 120` (`51.7%`) | Hand corrects on descent (`2.43 -> 0.83 cm` error), rim blocks drop (`51 -> 43`), and total pass reaches `51.7%` |
| **Exp 07 (`exp07_dart_full`)** | Full source + target DART demos (`data/sim_demos_v2_dart_full.h5`): source DART down to `dz = 3.5 cm` + target carry/lower DART | `118 / 120` (`98.3%`) | `104 / 120` (`86.7%`) | `40 / 60` (`66.7%`) | `26 / 60` (`43.3%`) | `66 / 120` (`55.0%`) | Target release error drops (`1.11 -> 0.90 cm`), target placement failures drop `-35%` (`20 -> 13`), `Task 1` jumps `42.5% -> 62.5%`, and total pass reaches `55.0%` |
| **Exp 08 (`exp08_dart_full_900`)** | **`3x` demo scale (`900` full-DART demos, `300/task`, `40` epochs) + test-time `obs_dropout` MC averaging (`obs_dropout_mc_k = 8`)** | **`120 / 120` (`100.0%`)** | **`117 / 120` (`97.5%`)** | **`57 / 60` (`95.0%`)** | **`50 / 60` (`83.3%`)** | **`107 / 120` (`89.2%`)** (`109 / 120 = 90.8%` at `k=4`) | **Eliminates `obs_dropout` train/eval shift (`34/120` at `k=0` $\to$ `107..109/120` at `k=4..16`), nearly doubles unseen test pass (`43.3% -> 83.3%..88.3%`), and lifts all 3 tasks to `85%..95%`** |

***

## Experiment 01: Clean Baseline (`exp01_clean_lags32`)

* **Commit / Checkpoint**: `fc826a0` | `checkpoints/exp01_clean_lags32/best_policy.pt`

### 1. Hypothesis
In early prototype runs, aggressive frame trimming accidentally deleted the frames where the robot opens its gripper and retracts upward at the end of a task. We hypothesized that restoring the complete 8-segment trajectory (`approach -> descend -> clamp -> lift -> transit -> lower -> release -> retract`) and giving the policy a 5-frame history of its recent joint positions (`lags = (0, 4, 8, 16, 32)`) would let the robot complete the full pick-and-place cycle.

### 2. Changes
* **Dataset ([`training/dataset.py`](./training/dataset.py))**: Kept all grasp, release, and retraction frames (`43,131` steps across `300` demos; `150` clean + `150` domain-randomized) and removed all hardcoded step-index overrides.
* **Model ([`training/config.py`](./training/config.py), [`training/model.py`](./training/model.py))**: Added 5-frame joint history (`30D` input: current `6D` joints + `4` past joint deltas) and fixed `SinusoidalTimeEmbedding` (removed `* 1000` frequency aliasing).
* **Evaluation ([`evaluation/evaluator.py`](./evaluation/evaluator.py))**: Evaluated `120` closed-loop rollouts (`60` train seeds + `60` unseen test seeds) with `max_steps = 280` (`14 s` at `20 Hz`).

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 01 (`8x8` Grid + `30D` History) |
| :--- | :---: |
| **Step-0 Camera Sensitivity** (`1.0` = follows camera) | `0.10` (blind to object position) |
| **Reached Object (`< 2.0 cm`)** | `27 / 120 (22.5%)` |
| **Lifted Object (`> 2.0 cm`)** | `3 / 120 (2.5%)` |
| **Task Success (`Train / Test / Total`)** | `0/60 (0.0%)` / `1/60 (1.7%)` / **`1/120 (0.8%)`** |

* **What worked**: Whenever the initial reach happened to land on the object (such as test seed `test_t0_s9013`), the robot grasped it, lifted it `13.5 cm`, carried it into the bowl, opened its gripper, and retracted cleanly.
* **What failed (The Copycat Problem)**: Swapping camera images vs. swapping joint history between a far-left object demo and a far-right object demo showed that the policy was **ignoring the cameras** and simply extrapolating its own past joint velocity. In addition, the `8x8` `SpatialSoftmax` layer (`temperature = 1.0` after `GroupNorm + SiLU`) washed out `90%` of the object signal into background cells.

### 4. What to Try Next
Remove the joint velocity history (`proprio_history_lags = (0,)`) so the policy cannot coast on past motion, and sharpen the camera encoder (`16x16` spatial grid, no pre-softmax `SiLU`, `temperature = 0.1`).

***

## Experiment 02a: Sharp `16x16` Encoder + Single-Frame Proprioception (`exp02a_sharp_nohist`)

* **Commit / Checkpoint**: `95b8200` | `checkpoints/exp02a_sharp_nohist/best_policy.pt`

### 1. Hypothesis
Removing multi-step joint history (`lags = (0,)`) and sharpening the camera keypoint extractor (`16x16` grid, `temperature = 0.1`, no pre-softmax `SiLU`) will prevent the policy from extrapolating joint velocity and force it to read the 3 cameras from `Step 00`.

### 2. Changes
* **Vision Encoder ([`training/model.py`](./training/model.py))**: Changed `conv4` from `8x8` (`stride=2`) to `16x16` (`stride=1`), removed `SiLU` before `SpatialSoftmax2d`, and set `init_temperature = 0.1`.
* **Proprioception ([`training/config.py`](./training/config.py))**: Set `proprio_history_lags = (0,)` (single-frame `6D` joint positions only).

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 01 (`8x8` + `30D` History) | **Exp 02a (`16x16` + `6D` Single-Frame)** |
| :--- | :---: | :---: |
| **Step-0 Camera Sensitivity** (`1.0` = follows camera) | `0.10` | **`0.34` (`3.4x` higher)** |
| **`overhead_cam` Linear Probe $R^2$ (`pan` angle / `shoulder` depth)** | `<= 0.00` / `<= 0.00` | **`+0.66`** / `+0.02` |
| **Reached Object (`< 2.0 cm`)** | `27 / 120 (22.5%)` | `27 / 120 (22.5%)` |
| **Lifted Object (`> 2.0 cm`)** | `3 / 120 (2.5%)` | **`8 / 120 (6.7%)`** (`6/60` on Test vs. `2/60`) |
| **Task Success (`Train / Test / Total`)** | `0/60` / `1/60` / `1/120 (0.8%)` | `0/60` / `1/60` / **`1/120 (0.8%)`** |

* **What worked**: Camera sensitivity jumped `3.4x` at `Step 00` (`0.10 -> 0.34`) and reached `0.78` during carry. The overhead camera now predicts horizontal target angle (`pan`) with $R^2 = +0.66$, and held-out test lifts tripled (`2/60 -> 6/60`).
* **What failed**: Forward reach depth (`shoulder` $R^2 = +0.02$) was still near zero. With 4 objects moving simultaneously on the table, `300` demonstrations are too few distinct layouts for end-to-end action loss alone to teach the CNNs exact 2D object coordinates.

### 4. What to Try Next
Pretrain the camera encoders on `3,000` rendered tabletop layouts with ground-truth 2D object positions (`Stage 1`), and keep an auxiliary object-position loss active during policy training (`Stage 2`).

***

## Experiment 02b: Stage 1 Vision Pretraining + Auxiliary Position Loss (`exp02b_pretrain_loc`)

* **Commit / Checkpoint**: `1a4b8bf` | `checkpoints/exp02b_pretrain_loc/best_policy.pt`

### 1. Hypothesis
Pretraining the 3 camera CNNs on `3,000` labeled tabletop scenes (`Stage 1`) to predict the `12D` `(X, Y)` coordinates of `[source, target, pen_holder, cup, bowl, rubiks_cube]`, and co-supervising training-only position heads alongside Flow Matching (`Stage 2`), will teach the vision encoders sub-centimeter object localization (`pan` and `shoulder` depth) with zero overhead at inference.

### 2. Changes
* **Stage 1 Vision Pretraining ([`training/pretrain_vision.py`](./training/pretrain_vision.py))**: Rendered `3,000` labeled 3-camera scenes (`data/loc_layouts_3000.npz`) and pretrained the camera encoders and `12D` position heads for `3,000` steps.
* **Stage 2 Co-Supervision ([`training/model.py`](./training/model.py), [`training/flow_matching.py`](./training/flow_matching.py))**: Added training-only `12D` position heads co-supervised during Flow Matching ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + 0.5 \cdot \mathcal{L}_{\text{pos}}$). Kept `z_fused` (`192D`) unchanged and ignored the position heads at inference.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 02a (No Position Supervision) | **Exp 02b (Stage 1 + Aux Position Loss)** |
| :--- | :---: | :---: |
| **Camera Object Localization Error (`tp` / `ov`)** | N/A | **`0.55 cm` / `0.60 cm`** ($R^2 = +0.99$) |
| **Step-0 Camera Sensitivity** (`1.0` = follows camera) | `0.34` | **`0.89`** |
| **Reached Object (`< 2.0 cm`)** | `27 / 120 (22.5%)` | **`107 / 120 (89.2%)`** (`mean XY = 0.79 cm`) |
| **Lifted Object (`> 2.0 cm`)** | `8 / 120 (6.7%)` | **`65 / 120 (54.2%)`** |
| **Task Success (`Train / Test / Total`)** | `0/60` / `1/60` / `1/120 (0.8%)` | **`9/60 (15.0%)` / `5/60 (8.3%)` / `14/120 (11.7%)`** |

* **What worked**: 2D object localization is solved (`0.55 to 0.60 cm` error). Reaching within `2.0 cm` of the source object jumped from `22.5%` to **`89.2%` (`107/120`)**, lifting jumped from `6.7%` to **`54.2%` (`65/120`)**, and strict task pass jumped from `1/120` to **`14/120`**.
* **What failed**: `MultiCameraAttention` used a frozen random `ProprioMLP` query during Stage 1 and then had to re-learn joint-conditioned attention weights on only `300` demos in Stage 2, hurting held-out test accuracy (`5/60` test vs. `9/60` train).

### 4. What to Try Next
Remove `MultiCameraAttention` and `aux_fused_pos_head` completely so the 3 camera tokens (`[tok_tp, tok_ov, tok_wr] = 96D`) are concatenated directly into `z_fused` (`192D`).

***

## Experiment 02c: Direct Camera Concatenation Without Attention (`exp02c_no_attn`)

* **Commit / Checkpoint**: `f728d47` | `checkpoints/exp02c_no_attn/best_policy.pt`

### 1. Hypothesis
Since each camera CNN already outputs a clean `32D` token with `~0.6 cm` localization accuracy, deleting `MultiCameraAttention` and concatenating the three camera tokens (`[tok_third_person, tok_overhead, tok_wrist] = 96D`) directly with `z_prop` (`64D`) and `z_task` (`32D`) will remove Stage 1-to-Stage 2 query shift and improve test generalization.

### 2. Changes
* **Model ([`training/model.py`](./training/model.py))**: Deleted `MultiCameraAttention` and `aux_fused_pos_head` (`-33,696` parameters). Concatenated `[tok_tp, tok_ov, tok_wr]` (`96D`) directly into `z_fused` (`192D`), and made Stage 1 strictly vision-only (`encode_vision_tokens`).

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 02b (With Camera Attention) | **Exp 02c (Direct Concatenation)** |
| :--- | :---: | :---: |
| **Camera Object Localization Error (`tp` / `ov` / `wr`)** | `0.55 cm` / `0.60 cm` / `2.12 cm` | **`0.56 cm` / `0.72 cm` / `1.90 cm`** |
| **Reached Object (`< 2.0 cm`)** | `107 / 120 (89.2%)` | `105 / 120 (87.5%)` |
| **Centered Grasp (`< 1.5 cm` at close)** | N/A | **`60 / 120 (50.0%)`** |
| **Lifted Object (`> 2.0 cm`)** | `65 / 120 (54.2%)` | **`75 / 120 (62.5%)`** |
| **Task Success (`Train / Test / Total`)** | `9/60 (15.0%)` / `5/60 (8.3%)` / `14/120 (11.7%)` | **`11/60 (18.3%)` / `16/60 (26.7%)` / `27/120 (22.5%)`** |

* **What worked**: Held-out test pass rate more than tripled (`5/60 -> 16/60`), total task success nearly doubled (`14/120 -> 27/120`), and object lifts rose to `75/120 (62.5%)` with a simpler architecture.
* **What failed**: Inspecting the `93` failed rollouts alongside the `300` `v1` demos (`data/sim_demos.h5`) exposed 3 flaws in the expert demonstrations:
  1. **1-frame V-bottom grasp**: `v1` demos touch the grasp bottom for only `1` frame before immediately rising.
  2. **Fixed close clock**: Every `v1` demo closes the gripper at exact step `42`, teaching the policy to close by step count even when the hand is still high or off-center (`30` grasp failures).
  3. **Rising while clamped at release**: At the target, `v1` demos pull upward `+1.8 cm` for `8` steps while the jaws are still clamped around the placed object, causing the policy to freeze over the target without opening (`7` episodes).

### 4. What to Try Next
Regenerate the `300` simulation demos on the exact same seeds (`data/sim_demos_v2.h5`) with a flat-bottom pause during grasp, stationary open-in-place release at the target, and $\pm 20\%$ segment timing jitter.

***

## Experiment 03: `v2` Demonstrations with Flat-Bottom Grasp & Open-in-Place Release (`exp03_demo_v2`)

* **Commit / Checkpoint**: `a935e0a` | `checkpoints/exp03_demo_v2/best_policy.pt`

### 1. Hypothesis
Replacing the `v1` demonstrations with `v2` trajectories (`data/sim_demos_v2.h5`) that pause at the bottom of the grasp to close in place, pause at the target to open in place before rising, and vary segment speed by $\mathcal{U}(0.8, 1.2)$ per episode will stop the robot from closing on a fixed timer and eliminate target release freezes.

### 2. Changes
* **Expert Demos ([`collection/sim_expert.py`](./collection/sim_expert.py))**: Added `trajectory_version = "v2"` and collected `data/sim_demos_v2.h5` (`300/300` pass rate on the same seeds) with `close_in_place` grasping, `open_in_place` release (`0` clamped-rise steps), and $\mathcal{U}(0.8, 1.2)$ timing jitter.
* **Training ([`training/dataset.py`](./training/dataset.py))**: Set `trim_stationary = False` for `v2` demos so intentional grasp/release dwell frames are preserved, and trained Stage 2 for `20` epochs from the `Exp 02c` Stage 1 vision checkpoint.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 02c (`v1` Demos, `20` Ep) | **Exp 03 (`v2` Demos, `20` Ep)** |
| :--- | :---: | :---: |
| **Reached Object (`< 2.0 cm`)** | `105 / 120 (87.5%)` | **`108 / 120 (90.0%)`** |
| **Centered Grasp (`< 1.5 cm` at close)** | `60 / 120 (50.0%)` | `55 / 120 (45.8%)` |
| **Lifted Object (`> 2.0 cm`)** | `75 / 120 (62.5%)` | `69 / 120 (57.5%)` |
| **Frozen Over Target Without Opening** | `7 / 120 (5.8%)` | **`3 / 120 (2.5%)` (`0/60` on Test)** |
| **Task Success (`Train / Test / Total`)** | `11/60 (18.3%)` / `16/60 (26.7%)` / `27/120 (22.5%)` | **`14/60 (23.3%)`** / `7/60 (11.7%)` / `21/120 (17.5%)` |

* **What worked**:
  - Target release freezes dropped by more than half (`7 -> 3`, and `0/60` on Test).
  - The gripper descended `~0.35 cm` deeper before closing and adapted its closing step (`step 43 to 74`) instead of firing on a fixed step-42 clock.
  - Overall accuracy (`21/120` vs. `27/120` pass, `45.8%` vs. `50.0%` centered grasp) was a statistical tie with `Exp 02c` within single-seed 20-epoch noise.
* **What failed (Follow-Up Diagnosis)**:
  1. **Open finger hits the object rim during descent**: In half the episodes (`45.8%` centered close), the open gripper arrived `~2.5 cm` off-center as it descended past the top rim of the cup/holder, so one open finger came down on the rim and knocked the object sideways.
  2. **The model was undertrained at 20 epochs**: Validation ODE error (`val_ode_mse = 0.00385`) was still dropping steeply at Epoch 20, step-to-step predictions had large sideways jitter, and centered grasp rates were equally low on training seeds (`45.0%`) and unseen test seeds (`46.7%`). *(An inference-only sweep of `temporal_ensemble_decay` across `0.05, 0.15, 0.30, 0.50` confirmed that default `0.05` is already optimal).*

### 4. What to Try Next
Train the exact `Exp 03` configuration **`3x` longer (`60` epochs instead of `20`)** so `val_ode_mse` converges and step-to-step prediction jitter is eliminated.

***

## Experiment 04: `60`-Epoch Training Schedule (`exp04_long_train_60ep`)

* **Commit / Checkpoint**: `9778d5a` | `checkpoints/exp04_long_train_60ep/best_policy.pt`

### 1. Hypothesis
In Experiment 03, `val_ode_mse` was still falling at Epoch 20 and the policy missed the grasp center just as often on training seeds (`45.0%`) as on test seeds (`46.7%`). We hypothesized that training the exact `Exp 03` setup for **`60` epochs (`3x` longer)** would let the flow model converge, cut fingertip prediction jitter, and substantially increase centered grasps and overall task success.

### 2. Changes
* **Training Schedule ([`training/config.py`](./training/config.py))**: Increased `epochs` from `20` to `60` (`240` stratified flow passes per sample) with cosine learning rate decay (`5e-4 -> 2.5e-5`). Kept the exact `Exp 03` model architecture, `v2` dataset (`data/sim_demos_v2.h5`), and Stage 1 vision checkpoint.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 03 (`20` Epochs) | **Exp 04 (`60` Epochs)** |
| :--- | :---: | :---: |
| **Validation ODE Error (`val_ode_mse`)** | `0.00385` | **`0.00220` (`-42.9%`)** |
| **Camera Object Localization Error (`tp` / `ov` / `wr`)** | `0.57 cm` / `0.65 cm` / `1.99 cm` | **`0.47 cm` / `0.52 cm` / `1.77 cm`** |
| **Reached Object (`< 2.0 cm`)** | `108 / 120 (90.0%)` | **`116 / 120 (96.7%)`** |
| **Centered Grasp (`< 1.5 cm` at close)** | `55 / 120 (45.8%)` (`2.54 cm` mean XY) | **`88 / 120 (73.3%)` (`1.17 cm` mean XY)** |
| **Lifted Object (`> 2.0 cm`)** | `69 / 120 (57.5%)` | **`103 / 120 (85.8%)`** |
| **Reached Target Zone (`< 6.0 cm`)** | `45 / 120 (37.5%)` | **`77 / 120 (64.2%)`** |
| **Task Success: Train Split (`60` eps)** | `14 / 60 (23.3%)` (`6 / 7 / 1`) | **`31 / 60 (51.7%)` (`10 / 9 / 12`)** |
| **Task Success: Test Split (`60` eps)** | `7 / 60 (11.7%)` (`0 / 4 / 3`) | **`22 / 60 (36.7%)` (`12 / 5 / 5`)** |
| **Task Success: Combined (`120` eps)** | `21 / 120 (17.5%)` | **`53 / 120 (44.2%, 2.52x higher)`** |

* **What worked**:
  - **Task success more than doubled** from `21/120 (17.5%)` to **`53/120 (44.2%)`** (`51.7%` on Train, `36.7%` on Test; `55.0%` on Task 0, `35.0%` on Task 1, `42.5%` on Task 2).
  - **Grasp centering and lift are largely solved**: Centered grasp close (`< 1.5 cm`) surged by **`+27.5` points** (`45.8% -> 73.3%`, cutting mean XY error at close from `2.54 cm` to `1.17 cm`), and **`85.8%` (`103/120`)** of episodes now successfully pick up the object.
  - **Target release freezes hit zero (`0/120`)**: Every episode that reached the target zone opened its gripper cleanly.
* **What still fails (`67` failed episodes out of `120`)**:
  1. **Shallow rim-blocked grasps (`45` shallow lifts + `13` failed grasps)**: Although `103/120` episodes lift the object, only `58` achieve a deep grasp (`81.0%` pass rate) while `45` catch only the top `1.5 to 2.0 cm` of the rim (`13.3%` pass rate) and later slip mid-air or bounce at release. Kinematic analysis showed that **`51 / 120` episodes are physically `BLOCKED`** (`26` on `moving_jaw_pad`, `23` on `fixed_jaw_pad`, `2` on neighbors): when the hand arrives `~1.4 cm` off-center at rim height (`dz = 4.5 cm`), one jaw pad strikes the top rim and stalls vertical descent (`command-vs-actual z gap >= 0.80 cm`) before the fingers close.
  2. **Lack of closed-loop descent correction**: Multi-seed rollout variance analysis (`21` scenes $\times$ `7` ODE noise draws) showed that `80%` of fingertip error is a deterministic per-scene bias and only `20%` is ODE sampling noise. The hand is already `~2.08 cm` off the expert path at hover, and because all `300` training demos follow perfect nominal paths, the policy never learned to steer back toward the object center during descent.

### 4. What to Try Next
1. Test whether widening the gripper opening during approach and descent (`Exp 05`) gives the pads enough clearance around the cup rim.
2. Add DART-style off-center hover perturbations paired with corrective action labels (`Exp 06`) so the policy learns to correct horizontal errors on the way down.

***

## Experiment 05: Wider Gripper Opening in Demonstrations (`exp05_wide_jaw`, Rejected)

* **Branch / Checkpoint**: `exp/05-wide-jaw` (`5c1cb76`) | `checkpoints/exp05_wide_jaw/best_policy.pt`

### 1. Hypothesis
In `v2` demos, the gripper descends at `0.65 -> 0.58` opening (`~51 mm` inner pad gap vs. `40 mm` cup diameter), leaving only `5.5 mm` of radial clearance per side. We hypothesized that widening the gripper during approach and descent (`0.85 -> 0.82` in `trajectory_version = "v3"`, `data/sim_demos_v3.h5`) and solving IK for the wider jaw midpoint would prevent rim hang-ups (`BLOCKED`) on both pads.

### 2. Changes
* **Expert Demos (`exp/05-wide-jaw`)**: Created `data/sim_demos_v3.h5` (`300/300` expert pass) with approach/hover opening `0.85`, descent `0.85 -> 0.82`, bottom dwell `0.82 -> 0.80`, and `n_close` stretched `8 -> 11` steps to match `v2` closing speed. Trained `60` epochs on Colab GPU with identical settings to `Exp 04`.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | **Exp 04 (`v2` Demos, `0.58` Jaw)** | Exp 05 (`v3` Demos, `0.82` Jaw) |
| :--- | :---: | :---: |
| **Validation ODE Error (`val_ode_mse`)** | **`0.00220`** | `0.00228` |
| **Reached Object (`< 2.0 cm`)** | **`116 / 120 (96.7%)`** | `96 / 120 (80.0%)` |
| **Centered Grasp (`< 1.5 cm` at close)** | **`88 / 120 (73.3%)`** (`0.81 cm` med XY) | `46 / 120 (38.3%)` (`1.88 cm` med XY) |
| **`BLOCKED` Grasps (`Total`: `Fixed / Moving / Nbr`)** | **`51`** (`23 / 26 / 2`) | `46` (`28 / 17 / 1`) |
| **Lifted Object (`> 2.0 cm`: `Deep / Shallow`)** | **`103 / 120`** (`58` deep / `45` shallow) | `74 / 120` (`49` deep / `25` shallow) |
| **Task Success (`Train / Test / Total`)** | **`31/60` / `22/60` / `53/120 (44.2%)`** | `21/60` / `21/60` / `42/120 (35.0%)` |

* **Why it failed (Single-Hinge Jaw Geometry)**: On the SO-ARM101 gripper, `fixed_jaw_pad` is bolted to the wrist housing while only `moving_jaw_pad` pivots outward. Widening the gripper to `0.82` shifts the geometric midpoint between the two pads `8 mm` toward the moving jaw. As the gripper closes from `0.80` to `0.05` during the clamp phase, the expert has to translate the entire arm `14.9 mm` sideways to keep the midpoint centered on the object. In closed-loop execution, small timing errors between arm translation and jaw closure caused the fixed jaw to side-swipe the object (`1_missed_xy_reach` jumped `4 -> 24` and `fixed_jaw_pad` blocks rose `23 -> 28`).
* **Decision**: **Rejected**. Kept `v2` jaw opening (`0.65 -> 0.58`) as the baseline geometry.

***

## Experiment 06: DART-Style Recovery Demonstrations (`exp06_dart`)

* **Commit / Checkpoint**: `6a9e31f` | `checkpoints/exp06_dart/best_policy.pt`

### 1. Hypothesis
In `Exp 04`, the hand arrived `2.08 cm` (median) off the expert path at the top of hover and failed to correct during descent (`1.13 cm` at close across all episodes; `1.57 -> 1.93 cm` on failed episodes) because every training demo showed an already-centered descent. We hypothesized that perturbing the executed demo path at hover by a random horizontal offset $\delta \sim \mathcal{U}(0, 2.5\text{ cm})$ while recording the clean `v2` expert commands as action labels (`data/sim_demos_v2_dart.h5`) would teach the policy closed-loop visual feedback during descent, shrinking close error, reducing `BLOCKED` rim collisions, and improving task pass rate.

### 2. Changes
* **DART Trajectory Perturbation ([`collection/sim_expert.py`](./collection/sim_expert.py), [`collection/collector.py`](./collection/collector.py))**:
  - Added `trajectory_version = "v2_dart"` and collected `data/sim_demos_v2_dart.h5` on the exact same `300` scene seeds (`300 / 300` first-attempt expert success rate).
  - For each episode, sampled direction $\theta \sim \mathcal{U}(0, 2\pi)$ and magnitude $\|\delta\| \sim \mathcal{U}(0, 2.5\text{ cm})$, solving IK at `(src_xy + delta, 0.145 m)` to obtain `dq_hover`.
  - **Executed path**: Ramped joint offset from `0` at start to `dq_hover` at hover (`median = 1.19 cm` off-path), held `w = 1.0` during upper descent (`dz >= 9.5 cm`), and decayed linearly to `w = 0.0` by `dz = 6.0 cm` so the physical arm settled within `< 0.1 cm` (`median = 0.084 cm`) by `dz = 5.5 cm`. Below `dz = 5.5 cm`, the executed path is 100% clean `v2`.
  - **Recorded action labels**: Stored the unperturbed clean `v2` expert action targets at every step (`median |label - exec| = 1.20 cm` at hover, `0.0000 cm` below `dz = 5.5 cm`).
* **Dataset & Training ([`training/dataset.py`](./training/dataset.py), [`training/config.py`](./training/config.py))**: Preserved stationary dwell frames (`trim_stationary = False` for `"v2_dart"`) and `lift_start_step` (`0.0` step difference vs. `v2`). Trained `60` epochs on Colab GPU from `checkpoints/exp02c_no_attn/pretrained_vision.pt`.

### 3. Results

#### 1. Primary Signal: Descent Trajectory Correction (`120` Benchmark Episodes vs. Clean Expert Path)

| Episode Group (`n`) | Hover Top (`z = 0.145 m`) Med / Mean `XY` | Pinch `dz = 6.0 cm` Med / Mean `XY` | Pinch `dz = 4.5 cm` Med / Mean `XY` | Grasp Close (`g < 0.48`) Med / Mean `XY` | Hover $\to$ Close Median Error Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Exp 06` ALL (`120`)** | **`2.43 / 3.07 cm`** | **`1.05 / 1.12 cm`** | **`1.02 / 1.12 cm`** | **`0.83 / 1.15 cm`** | **`-1.60 cm (-65.9%)`** |
| `Exp 04` ALL (`120`) | `2.08 / 3.11 cm` | `1.34 / 1.37 cm` | `1.22 / 1.34 cm` | `1.13 / 1.43 cm` | `-0.95 cm (-45.7%)` |
| **`Exp 06` PASS (`62`)** | **`2.48 / 2.96 cm`** | **`0.86 / 0.89 cm`** | **`0.83 / 0.86 cm`** | **`0.68 / 0.69 cm`** | **`-1.80 cm (-72.6%)`** |
| `Exp 04` PASS (`53`) | `1.84 / 2.41 cm` | `0.91 / 0.96 cm` | `0.87 / 0.93 cm` | `0.80 / 0.87 cm` | `-1.04 cm (-56.5%)` |
| **`Exp 06` FAIL (`58`)** | **`2.40 / 3.19 cm`** | **`1.32 / 1.38 cm`** | **`1.27 / 1.40 cm`** | **`1.31 / 1.65 cm`** | **`-1.09 cm (-45.4%)`** |
| `Exp 04` FAIL (`67`) | `2.54 / 3.67 cm` | `1.61 / 1.69 cm` | `1.57 / 1.67 cm` | `1.93 / 1.88 cm` | `-0.61 cm` *(diverges `+0.36 cm` below `dz = 4.5 cm`)* |

#### 2. Closed-Loop Funnel, `BLOCKED` Grasps & Task Success (`Exp 04` vs. `Exp 06`)

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 04 (`60` Ep, Nominal `v2`) | **Exp 06 (`60` Ep, `v2_dart` Recovery)** | Delta (`06 - 04`) |
| :--- | :---: | :---: | :---: |
| **Validation ODE Error (`val_ode_mse`)** | `0.00220` | **`0.00228`** | `+0.00008` |
| **Camera Object Localization Error (`tp` / `ov` / `wr`)** | `0.47 cm` / `0.52 cm` / `1.77 cm` | **`0.49 cm` / `0.55 cm` / `1.71 cm`** | Tied ($R^2 = +0.99$) |
| **Reached Object (`< 2.0 cm`)** | `116 / 120 (96.7%)` | **`116 / 120 (96.7%)`** | `0` |
| **Centered Grasp (`< 1.5 cm` at close)** | `88 / 120 (73.3%)` (`0.81 cm` med XY) | **`97 / 120 (80.8%)` (`0.74 cm` med XY)** | **`+9 (+7.5 pp)`** |
| **`BLOCKED` Grasps (`Total`: `Fixed / Moving / Nbr`)** | `51 / 120 (42.5%)` (`23 / 26 / 2`) | **`43 / 120 (35.8%)` (`23 / 19 / 1`)** | **`-8 (-15.7%)`** |
| **Lifted Object (`> 2.0 cm`: `Deep / Shallow`)** | `103 / 120` (`58` deep / `45` shallow) | **`103 / 120` (`63` deep / `40` shallow)** | **`+5` deep grasps** |
| **Reached Target Zone (`< 6.0 cm`)** | `77 / 120 (64.2%)` | **`93 / 120 (77.5%)`** | **`+16 (+13.3 pp)`** |
| **Task 0 (`pick_pen_holder_to_bowl`, `40` eps)** | `22 / 40 (55.0%)` (`1.20 cm` med close) | **`30 / 40 (75.0%)` (`0.82 cm` med close)** | **`+8 (+20.0 pp)`** |
| **Task 1 (`pick_cup_to_bowl`, `40` eps)** | `14 / 40 (35.0%)` (`1.48 cm` med close) | **`17 / 40 (42.5%)` (`0.90 cm` med close)** | **`+3 (+7.5 pp)`** |
| **Task 2 (`stack_cup_on_cube`, `40` eps)** | `17 / 40 (42.5%)` (`0.88 cm` med close) | `15 / 40 (37.5%)` (`0.73 cm` med close) | `-2 (-5.0 pp)` |
| **Task Success: Train Split (`60` eps)** | `31 / 60 (51.7%)` (`10 / 9 / 12`) | **`36 / 60 (60.0%)` (`15 / 12 / 9`)** | **`+5 (+8.3 pp)`** |
| **Task Success: Test Split (`60` eps)** | `22 / 60 (36.7%)` (`12 / 5 / 5`) | **`26 / 60 (43.3%)` (`15 / 5 / 6`)** | **`+4 (+6.7 pp)`** |
| **Task Success: Combined (`120` eps)** | `53 / 120 (44.2%)` | **`62 / 120 (51.7%)`** | **`+9 (+7.5 pp, +27 / -18 flips)`** |

* **What worked**:
  - **Closed-loop descent correction is active across all 3 tasks**: Median distance from the expert path shrinks by **`-65.9%`** during descent (`2.43 cm` at hover $\to$ `1.05 cm` at `dz = 6.0 cm` $\to$ `0.83 cm` at close). Even on `Task 1: pick_cup_to_bowl` (which stalled in `Exp 04`), median close error against the expert path fell from `1.48 cm` to **`0.90 cm` (`-39%`)**.
  - **Rim hang-ups drop and grasps deepen**: `BLOCKED` episodes dropped from `51` to **`43`** (`moving_jaw_pad` blocks fell `26 -> 19`), centered closes (`< 1.5 cm`) rose from `88` to **`97 / 120 (80.8%)`**, and deep grasps rose from `58` to **`63`**.
  - **Mid-air transit stalls (`3b`) were cut in half (`13 -> 6`)**: Deeper, better-centered grasps allowed **`93 / 120 (77.5%)`** of rollouts to carry the object into the `< 6.0 cm` target zone (up from `77 / 120 = 64.2%`), lifting total strict pass to **`62 / 120 (51.7%)`** (`75.0%` on Task 0, `42.5%` on Task 1, `37.5%` on Task 2).
* **Decision**: **Accepted as new baseline (`checkpoints/exp06_dart/best_policy.pt`)**.

### 4. What to Try Next
1. **Target placement and rim bounce (`4b + 4c = 25 / 58` failures)**: Now that `93 / 120 (77.5%)` of episodes reach the target zone, `25` episodes fail right at release (`18` bounce off the bowl/cube rim `4b`, and `7` nudge the target receptacle `4c`). Applying the same DART recovery principle to the **target hover and lower segment** can teach the policy to center over the bowl and Rubik's cube before releasing.
2. **Scale demonstration coverage on unseen test layouts (`60.0%` Train vs. `43.3%` Test)**: Expanding simulation demos beyond `100` per task can further close the `16.7`-point train-to-test gap on `Task 1` and `Task 2`.

***

## Experiment 07: Full Source + Target DART Recovery Demonstrations (`exp07_dart_full`)

* **Commit / Checkpoint**: `54bc98c` | `checkpoints/exp07_dart_full/best_policy.pt`

### 1. Hypothesis
In `Exp 06`, source-only DART stopped correcting at pinch height $dz = 5.5\text{ cm}$ (`1.5 cm` above the `4.0 cm` cup rim) and left the carry and target-lower segments unperturbed (`20` placement failures among the `82` episodes that reached within `3.0 cm` of the target). We hypothesized that:
1. Extending source DART deeper (`<= 0.2 cm` at $dz = 4.5\text{ cm}$ and `0` by $dz = 3.5\text{ cm}$) would provide corrective labels closer to the cup rim.
2. Adding an independent target DART perturbation $\delta_{\text{tgt}} \sim \mathcal{U}(0, 2.5\text{ cm})$ that ramps in during carry (after the lifted source clears `3.0 cm` above the table) to full at `hover_tgt` and decays linearly to `0` by `1.5 cm` above the final placement height (`place_z`), while recording clean `v2` action labels (`data/sim_demos_v2_dart_full.h5`), would teach closed-loop target centering and cut release errors and placement failures.

### 2. Changes
* **Full Source + Target DART Perturbation ([`collection/sim_expert.py`](./collection/sim_expert.py), [`collection/collector.py`](./collection/collector.py))**:
  - Added `trajectory_version = "v2_dart_full"` (`perturbation = "dart_full"`) and collected `data/sim_demos_v2_dart_full.h5` on the exact same `300` scene seeds (`300 / 300 = 100.0%` first-attempt expert success rate, `42,977` samples).
  - **Source DART**: Sampled $\delta_{\text{src}} \sim \mathcal{U}(0, 2.5\text{ cm})$, ramped from `0` at home to full at `hover_src` (`median = 1.19 cm`), capped linear decay during descent so physical offset is `<= 0.2 cm` (`median = 0.149 cm`) at $dz = 4.5\text{ cm}$ and `< 0.1 cm` (`median = 0.043 cm`) by $dz = 3.5\text{ cm}$, with `0.0000 cm` label-vs-executed difference below $dz = 3.5\text{ cm}$.
  - **Target DART**: Sampled independent $\delta_{\text{tgt}} \sim \mathcal{U}(0, 2.5\text{ cm})$, held `0` during lift until source lift $\ge 3.0\text{ cm}$, ramped to full at `hover_tgt` (`median = 1.14 cm` off-path, `median |label - exec| = 1.13 cm`), and decayed linearly during lower to `< 0.1 cm` (`median = 0.036 cm`) by `src_bottom_above_place_cm = 1.5 cm`, with `0.0000 cm` label-vs-executed difference below `1.5 cm`.
* **Dataset & Training ([`training/dataset.py`](./training/dataset.py), [`training/config.py`](./training/config.py))**: Preserved stationary dwell frames (`trim_stationary = False` for `"v2_dart_full"`). Trained `60` epochs on Colab GPU (`seed 42`) from `checkpoints/exp02c_no_attn/pretrained_vision.pt`.

### 3. Results

#### 1. Target Placement Correction (`hover_tgt` $\to$ `release`, Episodes Reaching `<= 3.0 cm` of Target)

| Group / Task | Split | `Exp 06` (`hover_tgt` $\to$ `release`, `median (p75)`) | **`Exp 07` (`hover_tgt` $\to$ `release`, `median (p75)`)** | Release Error Change |
| :--- | :--- | :---: | :---: | :---: |
| **Overall** | **ALL** | `1.64 (2.59)` $\to$ `1.11 (1.58) cm` `[n=82]` | **`1.55 (2.09)` $\to$ `0.90 (1.38) cm` `[n=79]`** | **`-0.21 cm (-18.9%)`** |
| | **PASS** | `1.28 (1.83)` $\to$ `0.89 (1.28) cm` `[n=62]` | **`1.25 (1.75)` $\to$ `0.76 (1.07) cm` `[n=66]`** | **`-0.13 cm (-14.6%)`** |
| **`Task 0`** (`pen_holder -> bowl`) | **ALL** | `1.37 (1.86)` $\to$ `0.66 (1.18) cm` `[n=33]` | `1.38 (1.89)` $\to$ `0.90 (1.23) cm` `[n=27]` | `+0.24 cm` |
| **`Task 1`** (`cup -> bowl`) | **ALL** | `1.98 (2.85)` $\to$ `1.33 (2.15) cm` `[n=28]` | **`1.69 (2.92)` $\to$ `1.07 (1.40) cm` `[n=31]`** | **`-0.26 cm` (`p75 -34.9%`)** |
| **`Task 2`** (`cup -> cube`) | **ALL** | `1.29 (2.85)` $\to$ `1.31 (1.93) cm` `[n=21]` | **`1.09 (1.83)` $\to$ `0.75 (1.14) cm` `[n=21]`** | **`-0.56 cm (-42.7%)`** |

#### 2. Closed-Loop Funnel, Placement Failures & Task Success (`Exp 06` vs. `Exp 07`)

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 06 (`v2_dart`, Source Only) | **Exp 07 (`v2_dart_full`, Source + Target)** | Delta (`07 - 06`) |
| :--- | :---: | :---: | :---: |
| **Validation ODE Error (`val_ode_mse` on clean `v2`)** | `0.00234` | **`0.00247`** (`0.00234` on `v2_dart_full`) | `+0.00013` |
| **Reached Object (`< 2.0 cm`)** | `116 / 120 (96.7%)` | **`118 / 120 (98.3%)`** | **`+2 (+1.7 pp)`** |
| **Source Descent Correction (`hover -> close` med `XY`)** | `2.43` $\to$ `0.83 cm` | **`2.36` $\to$ `0.83 cm`** (`Task 1`: `0.90 -> 0.85 cm`) | Tied overall (`-0.05 cm` on `Task 1`) |
| **`BLOCKED` Grasps (`Total`: `Fixed / Moving / Nbr`)** | `43 / 120` (`23 / 19 / 1`) | `47 / 120` (`26 / 21 / 0`; `Task 1`: **`18 -> 15`**) | `+4` (`-3` on `Task 1`, `0` neighbor) |
| **Lifted Object (`> 2.0 cm`: `Deep / Shallow`)** | `103 / 120` (`63` deep / `40` shallow) | **`104 / 120`** (`63` deep / `41` shallow; `Task 1`: **`33 -> 38`**) | **`+1` overall (`+5` on `Task 1`)** |
| **Target-Arrival Conversion (`Pass / Reached <= 3 cm`)** | `62 / 82 (75.6%)` (`20` fails) | **`66 / 79 (83.5%)` (`13` fails)** | **`+7.9 pp (-35.0%` placement fails)** |
| **Target/Bystander Pushed or Tipped (`4c`)** | `7 / 120` (`6` at target `<= 3 cm`) | **`3 / 120` (`2` at target `<= 3 cm`)** | **`-4 (-57.1%)`** |
| **Rim Bounce / Source Tipped at Release (`4b`)** | `18 / 120` | **`14 / 120`** | **`-4 (-22.2%)`** |
| **Task 0 (`pick_pen_holder_to_bowl`, `40` eps)** | `30 / 40 (75.0%)` | `24 / 40 (60.0%)` | `-6 (-15.0 pp)` |
| **Task 1 (`pick_cup_to_bowl`, `40` eps)** | `17 / 40 (42.5%)` | **`25 / 40 (62.5%)`** | **`+8 (+20.0 pp)`** |
| **Task 2 (`stack_cup_on_cube`, `40` eps)** | `15 / 40 (37.5%)` | **`17 / 40 (42.5%)`** | **`+2 (+5.0 pp)`** |
| **Task Success: Train Split (`60` eps)** | `36 / 60 (60.0%)` | **`40 / 60 (66.7%)`** | **`+4 (+6.7 pp)`** |
| **Task Success: Test Split (`60` eps)** | `26 / 60 (43.3%)` | **`26 / 60 (43.3%)`** | `0 (0.0 pp)` |
| **Task Success: Combined (`120` eps)** | `62 / 120 (51.7%)` | **`66 / 120 (55.0%)`** | **`+4 (+3.3 pp, +17 / -13 flips)`** |

* **What worked**:
  - **Target descent actively centers the held object before release**: Among episodes reaching within `3.0 cm` of the target, median release error dropped from `1.11 cm` to **`0.90 cm`** (`0.89 -> 0.76 cm` on pass episodes). On `Task 2` (`stack_cup_on_cube`), where `Exp 06` had zero correction during target lower (`1.29 -> 1.31 cm`), `Exp 07` cuts error by **`-42.7%`** (`1.09 -> 0.75 cm`).
  - **Target placement failures dropped by `-35%` (`20 -> 13`)**: Target/bystander knockovers (`4c`) fell from `7` to **`3`**, rim bounces (`4b`) fell from `18` to **`14`**, and target-arrival conversion rose from `75.6%` to **`83.5%`**.
  - **Cup tasks (`Task 1 + Task 2`) surged by `+10` passes**: `Task 1` (`pick_cup_to_bowl`) jumped **`+20.0` points** (`17/40 = 42.5%` $\to$ **`25/40 = 62.5%`**, with lifted episodes rising `33 -> 38/40` and `BLOCKED` dropping `18 -> 15`), and `Task 2` rose from `37.5%` to **`42.5%`**.
* **Why `Task 0` dropped `-6` while `Task 1 + Task 2` gained `+10` (Rim Height Geometry)**:
  - Source DART parameterized $dz$ relative to the object center (`init_src_z`), decaying to `0` at $dz = 3.5\text{ cm}$. For the `4.0 cm` tall cup (`Task 1, 2`), jaw tips cross the top rim at `pinch dz = 3.16 cm`, so the correction finishes cleanly above the cup rim. For the `7.0 cm` tall pen holder (`Task 0`), jaw tips cross the top rim at **`pinch dz = 4.66 cm`**, where the fixed-$dz$ schedule still had up to `0.26 cm` of residual perturbation right at rim entry.
* **Decision**: **Accepted as baseline (`checkpoints/exp07_dart_full/best_policy.pt`, `66 / 120 = 55.0%`, later superseded by `Exp 08`)**.

### 4. What to Try Next
1. **Scale demonstration coverage (`66.7%` Train vs. `43.3%` Test)**: Increasing unique scene layouts from `100` to `300` per task (`900` total demos) will directly target the `23.4`-point train-to-test generalization gap.
2. **Reference source DART $dz$ to the object's top rim (`dz_rim = pinch_z - top_rim_z`)**: Decaying source DART to `0` at `1.5 cm` above each object's top rim (`dz = 5.0 cm` for the `7.0 cm` pen holder, `dz = 3.5 cm` for the `4.0 cm` cup) can remove residual perturbation at rim entry on `Task 0`.

***

## Experiment 08: Scaling Full DART to 900 Demos + Test-Time Dropout Averaging (`exp08_dart_full_900`)

* **Commit / Checkpoint**: `9aa1a36` | `checkpoints/exp08_dart_full_900/best_policy.pt`

### 1. Hypothesis
In `Exp 06` and `Exp 07`, training pass rose from `60.0%` to `66.7%`, while unseen test pass remained flat at `26 / 60 (43.3%)`. With only `100` demonstrations per task (`50` clean + `50` domain-randomized) covering a `19 cm x 34 cm` workspace and `3` objects (`6D` relative arrangement space), the average nearest-neighbor scene distance was `~4.6 cm`, leaving unseen test arrangements out-of-distribution. We hypothesized that:
1. Scaling full source + target DART demonstrations `3x` from `300` to **`900` episodes** (`300` per task: `150` `sim_clean` + `150` `sim_dr`, `129,212` samples in `data/sim_demos_v2_dart_full_900.h5`) and training for `40` epochs (`74,160` gradient steps, `2x` the steps of `Exp 07`) would cut inter-scene spacing below the `2.5 cm` DART recovery tube and close the train-to-test generalization gap.
2. When evaluating the `74,160`-step checkpoint, standard deterministic `.eval()` mode (`obs_dropout_mc_k = 0`) scored only `34 / 120 (28.3%)` despite `119 / 120` object reaches and lower validation loss (`0.00208` vs. `0.00247`). Diagnosing the `TaskConditionedVisionFlowPolicy` velocity head revealed that switching `self.obs_dropout = nn.Dropout(0.05)` on `self.obs_proj(z_fused)` ([`training/model.py`](./training/model.py)) from `.train()` to `.eval()` at test time induces a `-0.071` normalized (`-0.018 to -0.025 rad`) bias on `shoulder_pan` (`action[0]`), shifting the gripper `+0.8 to +1.2 cm` toward `+Y` and causing `63` moving-pad rim collisions. Keeping `obs_dropout` active at inference and averaging the predicted flow velocity over `k` independent dropout masks per Euler ODE step (`obs_dropout_mc_k = 8`) would eliminate this train/eval mismatch with negligible latency overhead.

### 2. Changes
* **Scaled Full-DART Dataset (`data/sim_demos_v2_dart_full_900.h5`)**:
  - Collected `900` episodes (`300` per task: `150` `sim_clean` + `150` `sim_dr`, `129,212` frame-action pairs) using `trajectory_version = "v2_dart_full"` (`perturbation = "dart_full"`).
  - First-attempt expert success rate across all `900` scenes was `895 / 900 (99.44%)` (`Task 0`: `99.67%`, `Task 1`: `99.33%`, `Task 2`: `99.33%`), with `100.0%` after replacing `5` failed seeds (`1,699` step gap before benchmark test seeds `9000+`).
  - Pre-flight geometry matched `Exp 07`: median off-path distance `1.257 cm` at `hover_src`, `0.157 cm` at `pinch dz = 4.5 cm`, `0.047 cm` at `pinch dz = 3.5 cm`, `1.209 cm` at `hover_tgt`, and `0.039 cm` at `1.5 cm` above placement (`0.0000 cm` label-vs-executed difference below both decay cutoffs).
* **Training ([`training/config.py`](./training/config.py))**: Trained `40` epochs (`74,160` steps, `batch_size = 64`, cosine LR) from `checkpoints/exp02c_no_attn/pretrained_vision.pt`, saving `checkpoints/exp08_dart_full_900/best_policy.pt` (epoch `40`, `val_ode_mse = 0.00208` on clean `v2`).
* **Test-Time `obs_dropout` Monte Carlo Velocity Averaging ([`training/flow_matching.py`](./training/flow_matching.py), [`evaluation/evaluator.py`](./evaluation/evaluator.py), [`evaluation/__main__.py`](./evaluation/__main__.py))**:
  - Added `obs_dropout_mc_k: int = 8` to [`TrainingConfig`](./training/config.py) and [`ConditionalFlowMatcher.sample_action_chunk`](./training/flow_matching.py).
  - When `obs_dropout_mc_k > 0`, `sample_action_chunk` computes visual and proprioceptive features (`encode_observations`) once in `.eval()` mode, enables only `policy.obs_dropout` in `.train()` mode, tiles the `512D` observation vector across a batch of `k` copies per ODE step, averages the `k` predicted flow velocities (`v.mean(dim=0, keepdim=True)`), and restores the policy to `.eval()` mode in a `finally` block.

### 3. Results

#### 1. Closed-Loop Benchmark Across `obs_dropout_mc_k` Settings (`120` Episodes: `60` Train + `60` Test)

| Metric (`120` Episodes: `60` Train + `60` Test) | `Exp 07` (`300` Demos, `k=0`) | `Exp 08` (`900` Demos, `k=0` Eval) | **`Exp 08` (`k=8`, `seed=0`, Default)** | **`Exp 08` (`k=4`, `seed=0`)** | **`Exp 08` (`k=16`, `seed=0`)** | **`Exp 08` (`k=8`, `seed=500`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation ODE Error (`val_ode_mse` on clean `v2`)** | `0.00247` | `0.00208` | **`0.00208`** | **`0.00208`** | **`0.00208`** | **`0.00208`** |
| **Reached Object (`< 2.0 cm`)** | `118 / 120 (98.3%)` | `119 / 120 (99.2%)` | **`120 / 120 (100.0%)`** | **`120 / 120 (100.0%)`** | **`120 / 120 (100.0%)`** | **`120 / 120 (100.0%)`** |
| **Centered at Grasp Close (`< 1.5 cm`)** | `92 / 120 (76.7%)` | `102 / 120 (85.0%)` | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** |
| **`BLOCKED` Grasps (`Total`: `Fixed / Moving / Nbr`)** | `47 / 120` (`26 / 21 / 0`) | `72 / 120` (`9 / 63 / 0`) | **`45 / 120` (`3 / 42 / 0`)** | **`43 / 120` (`2 / 41 / 0`)** | **`45 / 120` (`3 / 42 / 0`)** | **`49 / 120` (`4 / 45 / 0`)** |
| **Lifted Object (`>= 2.0 cm`)** | `104 / 120 (86.7%)` | `78 / 120 (65.0%)` | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** |
| **Reached Target Zone (`< 6.0 cm`)** | `89 / 120 (74.2%)` | `66 / 120 (55.0%)` | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** | **`117 / 120 (97.5%)`** |
| **Task 0 (`pick_pen_holder_to_bowl`, `40` eps)** | `24 / 40 (60.0%)` | `11 / 40 (27.5%)` | **`36 / 40 (90.0%)`** | **`36 / 40 (90.0%)`** | **`37 / 40 (92.5%)`** | **`38 / 40 (95.0%)`** |
| **Task 1 (`pick_cup_to_bowl`, `40` eps)** | `25 / 40 (62.5%)` | `12 / 40 (30.0%)` | **`37 / 40 (92.5%)`** | **`38 / 40 (95.0%)`** | **`38 / 40 (95.0%)`** | **`38 / 40 (95.0%)`** |
| **Task 2 (`stack_cup_on_cube`, `40` eps)** | `17 / 40 (42.5%)` | `11 / 40 (27.5%)` | **`34 / 40 (85.0%)`** | **`35 / 40 (87.5%)`** | **`33 / 40 (82.5%)`** | **`33 / 40 (82.5%)`** |
| **Task Success: Train Split (`60` eps)** | `40 / 60 (66.7%)` | `18 / 60 (30.0%)` | **`57 / 60 (95.0%)`** | **`56 / 60 (93.3%)`** | **`57 / 60 (95.0%)`** | **`58 / 60 (96.7%)`** |
| **Task Success: Test Split (`60` eps)** | `26 / 60 (43.3%)` | `16 / 60 (26.7%)` | **`50 / 60 (83.3%)`** | **`53 / 60 (88.3%)`** | **`51 / 60 (85.0%)`** | **`51 / 60 (85.0%)`** |
| **Task Success: Combined (`120` eps)** | `66 / 120 (55.0%)` | `34 / 120 (28.3%)` | **`107 / 120 (89.2%)`** | **`109 / 120 (90.8%)`** | **`108 / 120 (90.0%)`** | **`109 / 120 (90.8%)`** |

#### 2. Per-Step Inference Latency (`ode_steps = 5`)

| Device | `k = 0` (Deterministic Eval) | `k = 4` (MC Dropout) | **`k = 8` (MC Dropout Default)** | `k = 16` (MC Dropout) | `k = 8` vs. `k = 0` Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU (Single Process)** | `2.86 ms / step` (`349.8 Hz`) | `3.87 ms / step` (`258.3 Hz`) | **`4.01 ms / step` (`249.6 Hz`)** | `4.15 ms / step` (`241.2 Hz`) | **`+1.15 ms / step` (`1.40x`)** |
| **Apple MPS** | `4.48 ms / step` (`223.1 Hz`) | `5.62 ms / step` (`178.0 Hz`) | **`5.81 ms / step` (`172.1 Hz`)** | `6.68 ms / step` (`149.8 Hz`) | **`+1.33 ms / step` (`1.30x`)** |

* **What worked**:
  - **Root-cause resolution of the `obs_dropout` train/eval shift**: Because `encode_observations` runs once per step and only the lightweight 4-block `ResMLP` runs on a batch of `k = 8` dropout-masked observation vectors, `obs_dropout_mc_k = 8` removes `96%` of the deterministic `shoulder_pan` bias (`-0.071 -> -0.003` normalized) while adding just `+1.15 ms/step` on CPU (`249.6 Hz` vs. `20 Hz` control rate).
  - **Every task exceeds `82%` and overall pass reaches `89.2%..90.8%` (`107..109 / 120`)**: Across `k = 4, 8, 16` and independent seed offsets, `Exp 08` achieves `120 / 120 (100.0%)` object reaches, `117 / 120 (97.5%)` centered grasps (`< 1.5 cm`), `117 / 120 (97.5%)` object lifts (`>= 2.0 cm`), `117 / 120 (97.5%)` target-zone arrivals (`< 6.0 cm`), and `107..109 / 120` strict task completions.
  - **Held-out test generalization nearly doubles (`43.3% -> 83.3%..88.3%`)**: Unseen test split pass jumped from `26 / 60 (43.3%)` in `Exp 07` to **`50 / 60 (83.3%)`** at `k = 8` (`53 / 60 = 88.3%` at `k = 4`), confirming that `300` full-DART demos per task provide dense workspace coverage across all three manipulation tasks.
* **Decision**: **Accepted as new hero baseline (`checkpoints/exp08_dart_full_900/best_policy.pt` with `obs_dropout_mc_k = 8`, `107 / 120 = 89.2%` at `k=8` and `109 / 120 = 90.8%` at `k=4`)**.

### 4. What to Try Next
1. **Remove or replace `obs_dropout` during training so `k = 0` matches `k = 8`**: Retraining without `obs_dropout` on `obs_proj(z_fused)` (or folding the dropout expectation into the layer norm calibration) can achieve the same `~90%` pass rate in single-sample deterministic `.eval()` mode (`k = 0`).
2. **Audit the remaining `11..13 / 120` failure episodes (`3` grasp/lift misses + `8..10` target placement bounces)**: Inspecting the `3` unlifted episodes and the `8..10` episodes that reach the target zone (`117 / 120`) but miss final placement criteria can push strict pass above `93%..95%`.

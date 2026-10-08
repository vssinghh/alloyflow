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
| **Exp 06 (`exp06_dart`)** | Off-center source hover recovery demos (`data/sim_demos_v2_dart.h5`, up to `2.5 cm` offset) | `116 / 120` (`96.7%`) | `103 / 120` (`85.8%`) | `36 / 60` (`60.0%`) | `26 / 60` (`43.3%`) | `62 / 120` (`51.7%`) | Hand steers back toward object center on descent (`2.43 -> 0.83 cm` error) and lifts pass rate to `51.7%` |
| **Exp 07 (`exp07_dart_full`)** | Source + target recovery demos (`data/sim_demos_v2_dart_full.h5`, `300` total) | `118 / 120` (`98.3%`) | `104 / 120` (`86.7%`) | `40 / 60` (`66.7%`) | `26 / 60` (`43.3%`) | `66 / 120` (`55.0%`) | Target release error drops (`1.11 -> 0.90 cm`), placement misses drop `35%`, and `Task 1` jumps `42.5% -> 62.5%` |
| **Exp 08 (`exp08_dart_full_900`)** | `3x` dataset scale (`900` full-DART demos, `300/task`) + test-time dropout averaging | `120 / 120` (`100.0%`) | `117 / 120` (`97.5%`) | `57 / 60` (`95.0%`) | `50 / 60` (`83.3%`) | `107 / 120` (`89.2%`) | `900` demos + fixing velocity-head dropout shift nearly doubles unseen test pass (`43.3% -> 83.3%`) |
| **Exp 08b (`exp08b_cooldown_k0`)** | **Zero-dropout flow velocity head (`dropout = 0.0`, single-pass `.eval()`)** | **`120 / 120` (`100.0%`)** | **`118 / 120` (`98.3%`)** | **`58 / 60` (`96.7%`)** | **`54 / 60` (`90.0%`)** | **`112 / 120` (`93.3%`)** | **Removing dropout eliminates train/eval mismatch, cuts rim blocks in half (`45 -> 21`), and reaches `93.3%` at `350 Hz` CPU** |

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
  1. **Shallow rim-blocked grasps (`45` shallow lifts + `13` failed grasps)**: Although `103/120` episodes lift the object, `58` achieve a deep grasp (`81.0%` pass rate) while `45` catch only the top `1.5 to 2.0 cm` of the rim (`13.3%` pass rate) and later slip mid-air or bounce at release. In **`51 / 120` episodes**, when the hand arrives `~1.4 cm` off-center at rim height (`dz = 4.5 cm`), one jaw pad hits the top rim and physically blocks downward travel before the fingers close.
  2. **No closed-loop correction during descent**: At the top of hover, the hand is already `~2.08 cm` off the expert path. Because all `300` training demos follow perfectly centered paths, the policy never learned to steer back toward the object center on the way down.

### 4. What to Try Next
1. Test whether widening the gripper opening during approach and descent (`Exp 05`) gives the pads enough clearance around the cup rim.
2. Add DART-style off-center hover perturbations paired with corrective action labels (`Exp 06`) so the policy learns to correct horizontal errors on the way down.

***

## Experiment 05: Wider Gripper Opening in Demonstrations (`exp05_wide_jaw`, Rejected)

* **Branch / Checkpoint**: `exp/05-wide-jaw` (`5c1cb76`) | `checkpoints/exp05_wide_jaw/best_policy.pt`

### 1. Hypothesis
In `v2` demos, the gripper descends at `0.65 -> 0.58` opening (`~51 mm` inner pad gap vs. `40 mm` cup diameter), leaving only `5.5 mm` of radial clearance per side. We hypothesized that widening the gripper during approach and descent (`0.85 -> 0.82` in `trajectory_version = "v3"`, `data/sim_demos_v3.h5`) and solving IK for the wider jaw midpoint would prevent rim hang-ups on both pads.

### 2. Changes
* **Expert Demos (`exp/05-wide-jaw`)**: Created `data/sim_demos_v3.h5` (`300/300` expert pass) with approach/hover opening `0.85`, descent `0.85 -> 0.82`, bottom dwell `0.82 -> 0.80`, and `n_close` stretched `8 -> 11` steps to match `v2` closing speed. Trained `60` epochs on Colab GPU with identical settings to `Exp 04`.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | **Exp 04 (`v2` Demos, `0.58` Jaw)** | Exp 05 (`v3` Demos, `0.82` Jaw) |
| :--- | :---: | :---: |
| **Validation ODE Error (`val_ode_mse`)** | **`0.00220`** | `0.00228` |
| **Reached Object (`< 2.0 cm`)** | **`116 / 120 (96.7%)`** | `96 / 120 (80.0%)` |
| **Centered Grasp (`< 1.5 cm` at close)** | **`88 / 120 (73.3%)`** (`0.81 cm` med XY) | `46 / 120 (38.3%)` (`1.88 cm` med XY) |
| **Rim-Blocked Grasps** | **`51 / 120`** | `46 / 120` |
| **Lifted Object (`> 2.0 cm`: `Deep / Shallow`)** | **`103 / 120`** (`58` deep / `45` shallow) | `74 / 120` (`49` deep / `25` shallow) |
| **Task Success (`Train / Test / Total`)** | **`31/60` / `22/60` / `53/120 (44.2%)`** | `21/60` / `21/60` / `42/120 (35.0%)` |

* **Why it failed (Single-Hinge Jaw Geometry)**: On the SO-ARM101 gripper, the fixed jaw is bolted to the wrist housing while only the moving jaw pivots outward. Widening the gripper to `0.82` shifts the midpoint between the two pads `8 mm` toward the moving jaw. As the gripper closes from `0.80` to `0.05` during the clamp phase, the expert has to translate the entire arm `14.9 mm` sideways to keep the midpoint centered on the object. In closed-loop execution, small timing errors between arm translation and jaw closure caused the fixed jaw to side-swipe the object (missed horizontal reaches jumped `4 -> 24`).
* **Decision**: **Rejected**. Kept `v2` jaw opening (`0.65 -> 0.58`) as the baseline geometry.

***

## Experiment 06: Off-Center Hover Recovery Demonstrations (`exp06_dart`)

* **Commit / Checkpoint**: `6a9e31f` | `checkpoints/exp06_dart/best_policy.pt`

### 1. Hypothesis
In `Exp 04`, the hand arrived `2.08 cm` off the expert path at hover and failed to steer back toward the center during descent because every training demo showed an already-centered descent. We hypothesized that shifting the executed demo path at hover by a random horizontal offset (`0 to 2.5 cm`) while recording the clean centered `v2` joint targets as action labels (`data/sim_demos_v2_dart.h5`) would teach the policy to correct horizontal errors on the way down.

### 2. Changes
* **DART Recovery Demos ([`collection/sim_expert.py`](./collection/sim_expert.py))**: Added `trajectory_version = "v2_dart"` (`data/sim_demos_v2_dart.h5`, `300/300` expert pass). Each episode shifts the executed hover position by a random horizontal offset up to `2.5 cm` and smoothly blends back to the centered path by `5.5 cm` above the object center while saving the unperturbed clean `v2` targets as action labels.
* **Training**: Trained `60` epochs on Colab GPU from the `Exp 02c` Stage 1 vision checkpoint.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 04 (Nominal `v2` Demos) | **Exp 06 (Source DART Recovery)** |
| :--- | :---: | :---: |
| **Hover $\to$ Close Path Error (`Med XY`)** | `2.08 cm` $\to$ `1.13 cm` (`-45.7%`) | **`2.43 cm` $\to$ `0.83 cm` (`-65.9%`)** |
| **Reached Object (`< 2.0 cm`)** | `116 / 120 (96.7%)` | **`116 / 120 (96.7%)`** |
| **Centered Grasp (`< 1.5 cm` at close)** | `88 / 120 (73.3%)` | **`97 / 120 (80.8%)`** |
| **Rim-Blocked Grasps** | `51 / 120 (42.5%)` | **`43 / 120 (35.8%)`** |
| **Lifted Object (`> 2.0 cm`: `Deep / Shallow`)** | `103 / 120` (`58` deep / `45` shallow) | **`103 / 120` (`63` deep / `40` shallow)** |
| **Reached Target Zone (`< 6.0 cm`)** | `77 / 120 (64.2%)` | **`93 / 120 (77.5%)`** |
| **Task Success (`Train / Test / Total`)** | `31/60` / `22/60` / `53/120 (44.2%)` | **`36/60 (60.0%)` / `26/60 (43.3%)` / `62/120 (51.7%)`** |

* **What worked**:
  - **Active descent correction**: Median distance from the expert path shrinks by **`65.9%`** during descent (`2.43 cm` at hover down to `0.83 cm` at close).
  - **Fewer rim collisions and deeper grasps**: Centered closes rose from `88` to **`97 / 120`**, rim-blocked grasps fell from `51` to **`43`**, target-zone arrivals jumped from `77` to **`93 / 120`**, and total task pass reached **`62 / 120 (51.7%)`** (`75.0%` on Task 0).
* **What failed**: Among the `93` episodes that reached the target zone, `25` failed during placement (`18` bounced off the bowl or cube rim and `7` nudged the target receptacle) because the carry and target-lower segments had no recovery perturbations.

### 4. What to Try Next
Extend DART recovery perturbations to the **target hover and lower segments** (`Exp 07`) so the policy also learns to center the held object over the bowl and Rubik's cube before releasing.

***

## Experiment 07: Source + Target DART Recovery Demonstrations (`exp07_dart_full`)

* **Commit / Checkpoint**: `54bc98c` | `checkpoints/exp07_dart_full/best_policy.pt`

### 1. Hypothesis
Adding an independent horizontal offset (`0 to 2.5 cm`) at target hover that decays to zero `1.5 cm` above the release height, and extending source descent recovery closer to the cup rim (`dz = 3.5 cm`), will teach the policy to center the held object over the target receptacle before opening the gripper.

### 2. Changes
* **Full Source + Target DART Demos ([`collection/sim_expert.py`](./collection/sim_expert.py))**: Added `trajectory_version = "v2_dart_full"` (`data/sim_demos_v2_dart_full.h5`, `300/300` expert pass). Applied independent `0 to 2.5 cm` horizontal offsets to both source hover (decaying to `0` by `dz = 3.5 cm`) and target hover (ramping in after the lifted object clears `3.0 cm` and decaying to `0` by `1.5 cm` above placement), while recording clean `v2` action targets.
* **Training**: Trained `60` epochs on Colab GPU from the `Exp 02c` Stage 1 vision checkpoint.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 06 (Source DART Only) | **Exp 07 (Source + Target DART)** |
| :--- | :---: | :---: |
| **Reached Object (`< 2.0 cm`)** | `116 / 120 (96.7%)` | **`118 / 120 (98.3%)`** |
| **Lifted Object (`> 2.0 cm`)** | `103 / 120 (85.8%)` | **`104 / 120 (86.7%)`** |
| **Target Hover $\to$ Release Error (`Med XY`)** | `1.64 cm` $\to$ `1.11 cm` | **`1.55 cm` $\to$ `0.90 cm` (`-18.9%`)** |
| **Target Placement Failures (`<= 3 cm` of target)** | `20` (`75.6%` conversion) | **`13` (`-35.0%`, `83.5%` conversion)** |
| **Task Pass by Task (`Task 0 / Task 1 / Task 2`)** | `30/40` / `17/40` / `15/40` | `24/40` / **`25/40`** / **`17/40`** |
| **Task Success (`Train / Test / Total`)** | `36/60 (60.0%)` / `26/60 (43.3%)` / `62/120 (51.7%)` | **`40/60 (66.7%)` / `26/60 (43.3%)` / `66/120 (55.0%)`** |

* **What worked**:
  - **Target centering before release**: Median target release error dropped from `1.11 cm` to **`0.90 cm`** (and by **`42.7%`** on `Task 2: stack_cup_on_cube`, from `1.31 cm` to `0.75 cm`), cutting placement failures by **`35%` (`20 -> 13`)**.
  - **Cup manipulation surged**: `Task 1 (pick_cup_to_bowl)` jumped **`+20` points** (`42.5% -> 62.5%`), `Task 2` rose to `42.5%`, and total pass reached **`66 / 120 (55.0%)`**.
* **What failed**: While training pass rose to `66.7%`, unseen test pass stayed flat at **`43.3%` (`26/60`)**. With only `100` demos per task across a `19 cm x 34 cm` workspace with 3 moving objects, average spacing between training layouts is `~4.6 cm`, leaving unseen test layouts outside the `2.5 cm` DART recovery tube.

### 4. What to Try Next
Scale the `v2_dart_full` dataset **`3x` from `300` to `900` demonstrations (`300` per task)** so training scenes densely cover the workspace and close the train-to-test gap.

***

## Experiment 08: Scaling Full DART to `900` Demos + Fixing Velocity-Head Dropout Shift (`exp08_dart_full_900`)

* **Commit / Checkpoint**: `9aa1a36` | `checkpoints/exp08_dart_full_900/best_policy.pt`

### 1. Hypothesis
1. Scaling `v2_dart_full` demonstrations `3x` from `300` to **`900` episodes** (`300` per task: `150` clean + `150` domain-randomized) will reduce the spacing between training scenes below the `2.5 cm` DART recovery radius and close the train-to-test generalization gap.
2. When evaluating the `900`-demo policy in standard `.eval()` mode, pass rate unexpectedly dropped to `34 / 120 (28.3%)` despite `119 / 120` object reaches and lower validation error (`0.00208` vs. `0.00247`). Inspecting the velocity head revealed that turning off `nn.Dropout(0.05)` at test time introduced a `-0.071` normalized bias on `shoulder_pan` (`~1 cm` sideways shift into the moving jaw pad). We hypothesized that averaging the predicted velocity across `8` dropout masks per ODE step at test time would remove this train/eval shift and unlock the true performance of the `900`-demo model.

### 2. Changes
* **Dataset (`data/sim_demos_v2_dart_full_900.h5`)**: Collected `900` full-DART demonstrations (`300` per task, `128,919` frames, `100%` expert pass) and trained for `40` epochs (`74,160` gradient steps, `2x` the steps of `Exp 07`).
* **Diagnostic Test-Time Dropout Averaging**: Evaluated the `Exp 08` checkpoint with `8` Monte Carlo dropout masks per Euler ODE step to eliminate the `obs_dropout` train-to-eval mean shift (before removing dropout altogether in `Exp 08b`).

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 07 (`300` Demos) | Exp 08 (`900` Demos, Raw `.eval()`) | **Exp 08 (`900` Demos, MC Dropout `k=8`)** |
| :--- | :---: | :---: | :---: |
| **Validation ODE Error (`val_ode_mse`)** | `0.00247` | `0.00208` | **`0.00208`** |
| **Reached Object (`< 2.0 cm`)** | `118 / 120 (98.3%)` | `119 / 120 (99.2%)` | **`120 / 120 (100.0%)`** |
| **Centered Grasp (`< 1.5 cm` at close)** | `92 / 120 (76.7%)` | `102 / 120 (85.0%)` | **`117 / 120 (97.5%)`** |
| **Lifted Object (`> 2.0 cm`)** | `104 / 120 (86.7%)` | `78 / 120 (65.0%)` | **`117 / 120 (97.5%)`** |
| **Reached Target Zone (`< 6.0 cm`)** | `89 / 120 (74.2%)` | `66 / 120 (55.0%)` | **`117 / 120 (97.5%)`** |
| **Task Pass by Task (`Task 0 / Task 1 / Task 2`)** | `24/40` / `25/40` / `17/40` | `11/40` / `12/40` / `11/40` | **`36/40 (90%)` / `37/40 (92.5%)` / `34/40 (85%)`** |
| **Task Success (`Train / Test / Total`)** | `40/60` / `26/60` / `66/120 (55.0%)` | `18/60` / `16/60` / `34/120 (28.3%)` | **`57/60 (95.0%)` / `50/60 (83.3%)` / `107/120 (89.2%)`** |

* **What worked**:
  - **Root cause of the `.eval()` drop identified and fixed**: Removing the `obs_dropout` train-to-eval shift eliminated `96%` of the `shoulder_pan` bias (`-0.071 -> -0.003`), lifting centered grasps to **`117 / 120 (97.5%)`** and object lifts to **`117 / 120 (97.5%)`**.
  - **Held-out test accuracy nearly doubled (`43.3% -> 83.3%`)**: Unseen test pass jumped from `26 / 60` in `Exp 07` to **`50 / 60 (83.3%)`** (`53 / 60 = 88.3%` at `k = 4`), and overall pass jumped from `55.0%` to **`89.2%` (`107 / 120`)** across all 3 tasks.

### 4. What to Try Next
Eliminate dropout (`dropout = 0.0`) from the flow velocity head so standard single-pass `.eval()` works directly without needing Monte Carlo dropout sampling at test time (`Exp 08b`).

***

## Experiment 08b: Zero-Dropout Flow Velocity Network & Input-Modality Shortcut Ablation (`exp08b_cooldown_k0`)

* **Commit / Checkpoint**: `c64dc1f` | `checkpoints/exp08b_cooldown_k0/best_policy.pt` (`5`-ep cooldown), `checkpoints/exp08_nodropout_scratch/best_policy.pt` (`40`-ep velocity head from scratch), & `checkpoints/exp08_nodropout_e2e/best_policy.pt` (Stage A + B zero-regularization ablation)

### 1. Hypothesis
1. **Stage B (`predict_velocity` neuron dropout)**: With `900` full-DART demos (`128,919` frames), domain randomization, and continuous Gaussian flow noise $x_0 \sim \mathcal{N}(0, I)$, the `9` neuron-dropout layers inside `predict_velocity` (`obs_dropout` + `4x ResMLPBlock`) are unnecessary and create a `LayerNorm + SiLU` variance shift between `.train()` and `.eval()`. Setting `dropout = 0.0` (either via a 5-epoch cooldown or training the velocity head from scratch for 40 epochs) will allow single-pass `.eval()` to match or beat Monte Carlo dropout at `350 Hz` CPU speed.
2. **Stage A (`extract_obs_features` input-modality regularizers)**: We also tested whether the Stage A input-modality regularizers (`proprio_noise_std = 0.02`, `proprio_drop_prob = 0.10`, `wrist_cam_drop_prob = 0.05`, `keypoint_noise = 0.01`) can be set to `0.0` during end-to-end training (`exp08_nodropout_e2e`), or whether they are required to prevent proprioceptive shortcut learning (causal confusion).

### 2. Changes
* **5-Epoch Zero-Dropout Cooldown (`checkpoints/exp08b_cooldown_k0/best_policy.pt`)**: Froze the `Exp 08` Stage A observation encoders in `.eval()` mode, set all `9` velocity-head dropout layers to `p = 0.0`, and fine-tuned the `Flow ResMLP` for `5` epochs (`lr = 5e-5 -> 1e-6`, `12.5 seconds` total).
* **40-Epoch Zero-Dropout Velocity Head From Scratch (`checkpoints/exp08_nodropout_scratch/best_policy.pt`)**: Reset all `778,592` parameters of `predict_velocity` to random initialization (`seed = 42`) and trained for `40` epochs (`lr = 5e-4 -> 2.5e-5`) with `dropout = 0.0` on the Stage A features (`proprio_noise_std = 0.02`, `proprio_drop_prob = 0.10`, `wrist_cam_drop_prob = 0.05`, `keypoint_noise = 0.01`).
* **End-to-End Zero-Regularization Ablation (`checkpoints/exp08_nodropout_e2e/best_policy.pt`)**: Trained both Stage A and Stage B end-to-end (`27` epochs + `5`-epoch cooldown to `lr = 1e-6`) with `dropout = 0.0` AND all Stage A input regularizers set to `0.0`.
* **Code Cleanup ([`training/config.py`](./training/config.py), [`training/flow_matching.py`](./training/flow_matching.py), [`evaluation/evaluator.py`](./evaluation/evaluator.py))**: Set `dropout = 0.0` permanently in `AlloyTrainConfig`, kept the Stage A anti-shortcut regularizers (`keypoint_noise = 0.01`, `proprio_noise_std = 0.02`, `proprio_drop_prob = 0.10`, `wrist_cam_drop_prob = 0.05`), and removed `obs_dropout_mc_k` across the codebase.

### 3. Results

| Metric (`120` Episodes: `60` Train + `60` Test) | Exp 08 (MC Dropout `k=8`) | **Exp 08b (`5`-Ep Cooldown, `dropout=0.0`)** | **Exp 08b (`40`-Ep Scratch, `dropout=0.0`)** | Exp 08 E2E Ablation (Stage A Noise/Drop `= 0.0`) |
| :--- | :---: | :---: | :---: | :---: |
| **Inference Mode & CPU Speed (`ode_steps = 5`)** | `8` MC passes (`250 Hz`) | **Single-pass `.eval()` (`350 Hz`)** | **Single-pass `.eval()` (`350 Hz`)** | Single-pass `.eval()` (`350 Hz`) |
| **Open-Loop Flow Loss (`cfm_loss`) / `val_ode_mse`** | `0.0524` / `0.00192` | `0.0439` / **`0.00151`** | `0.0435` / **`0.00172`** | `0.0377` / `0.00169` |
| **`obs_proj` Weight Norm on `overhead_cam`** | `5.86` | **`5.85`** | **`5.06`** | `4.75` (`-19.0%`) |
| **Reached Object (`< 2.0 cm`)** | `120 / 120 (100.0%)` | **`120 / 120 (100.0%)`** | **`120 / 120 (100.0%)`** | `120 / 120 (100.0%)` |
| **Centered Grasp (`< 1.5 cm` at close)** | `117 / 120 (97.5%)` | **`118 / 120 (98.3%)`** | **`117 / 120 (97.5%)`** | `107 / 120 (89.2%)` |
| **Rim-Blocked Grasps** | `45 / 120` | **`21 / 120` (`-53.3%`)** | **`36 / 120` (`-20.0%`)** | `34 / 120` |
| **Lifted Object (`> 2.0 cm`) & Mean Lift** | `117 / 120` (`+12.4 cm`) | **`118 / 120` (`+12.7 cm`)** | **`118 / 120` (`+12.6 cm`)** | `104 / 120` (`+9.7 cm`, `-3.0 cm`) |
| **Task Pass by Task (`Task 0 / Task 1 / Task 2`)** | `36/40` / `37/40` / `34/40` | **`38/40 (95%)` / `38/40 (95%)` / `36/40 (90%)`** | **`37/40 (92.5%)` / `37/40 (92.5%)` / `38/40 (95%)`** | `33/40` / `20/40` / `22/40` |
| **Task Success (`Train / Test / Total`)** | `57/60` / `50/60` / `107/120 (89.2%)` | **`58/60 (96.7%)` / `54/60 (90.0%)` / `112/120 (93.3%)`** | **`59/60 (98.3%)` / `53/60 (88.3%)` / `112/120 (93.3%)`** | `39/60 (65.0%)` / `36/60 (60.0%)` / `75/120 (62.5%)` |

* **What worked**:
  - **Both the 5-epoch cooldown and 40-epoch zero-dropout velocity head reach `112 / 120 (93.3%)` in standard `.eval()` mode**: Training `predict_velocity` from scratch with `dropout = 0.0` matches the `93.3%` total pass rate (`98.3%` Train, `88.3%` Test, and `95.0%` on `Task 2: stack_cup_on_cube`), proving that neuron dropout inside the flow ODE network is unnecessary once `900` full-DART demos are present.
  - **Rim-blocked grasps drop by half (`45 -> 21`) and inference runs at `350 Hz` on CPU**: Training and inference now use the exact same deterministic `.eval()` pass with zero test-time workarounds.
* **What failed in the `exp08_nodropout_e2e` ablation (and why Stage A regularizers must stay active)**:
  - Zeroing out the Stage A input-modality regularizers (`proprio_noise_std = 0.0`, `proprio_drop_prob = 0.0`, `wrist_cam_drop_prob = 0.0`, `keypoint_noise = 0.0`) produced the **lowest open-loop training loss** (`cfm_loss = 0.0377`, `val_ode_mse = 0.00169`) but dropped closed-loop rollout success from **`93.3%` to `62.5%`** (`50 / 120` at Epoch 27; `75 / 120` after 5-epoch cooldown).
  - **Root cause (Proprioceptive Causal Confusion)**: When `proprio` ($q_t$) is 100% noiseless and never dropped during Stage 2 training, the network minimizes open-loop MSE by extrapolating the minimum-jerk spline directly from $q_t$ and `wrist_cam`, reducing its `obs_proj` weight norm on `overhead_cam` by **`19.0%` (`5.86 -> 4.75`)**. In closed-loop physics, once the grasped cup or pen holder loads the wrist servo and occludes `wrist_cam`, the policy cuts its transit lift arc **`3.0 cm` short (`+9.7 cm` vs `+12.7 cm`)**, clipping the rim of the target bowl or Rubik's cube during placement. Keeping `proprio_noise_std = 0.02`, `proprio_drop_prob = 0.10`, `wrist_cam_drop_prob = 0.05`, and `keypoint_noise = 0.01` in Stage A while setting `dropout = 0.0` in Stage B prevents this shortcut and preserves the `112 / 120 (93.3%)` closed-loop pass rate.

### 4. What to Try Next
Evaluate real-robot (`SO-ARM101`) finetuning (`Mode 3`) and `50/50` co-training (`Mode 4`) starting from `checkpoints/exp08b_cooldown_k0/best_policy.pt`.



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
| **Exp 02a (`exp02a_sharp_nohist`)** | Un-blind the camera encoder and remove the joint velocity copycat shortcut | `16x16` spatial grid (`conv4` stride 1), removed pre-softmax `SiLU`, `temp=0.1`, single-frame `lags=(0,)` (`6D`) | `0 / 60` (`0.0%`) | `1 / 60` (`1.7%`) | **`6 / 60` (`10.0%`)** | **`0.34` (`3.4x` higher; `overhead_cam` pan $R^2 = +0.66$)** | Vision is now active and test lifts tripled (`2/60 -> 6/60`), but 300 demos are too few to learn accurate forward reach depth (`shoulder` $R^2 = +0.02$) |
| **Exp 02b (`exp02b_pretrain_loc`)** | Teach the camera encoders exact 2D object positions before policy training | Pretrain camera CNNs on rendered labeled tabletop layouts + auxiliary object position head | `Planned` | `Planned` | `Planned` | `Planned` | Pending run |

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
1. **Experiment 02b (Visual Localization Pretraining / Auxiliary Supervision)**: Train the camera encoders with direct supervision on rendered tabletop object positions (or an auxiliary training-only object position head) so `pan` and `shoulder` reach depth both achieve high $R^2$ from `Step 00` before the Flow Matching head is trained.

# AlloyFlow Experimental Log

This document records the controlled experimental runs for **AlloyFlow** on the 6-DoF SO-ARM101 multi-task benchmark (`Task 0: pick_pen_holder_to_bowl`, `Task 1: pick_cup_to_bowl`, `Task 2: stack_cup_on_cube`).

***

## Experiment 01: Clean Baseline (`exp01_clean_lags32`)

* **Branch**: `exp/clean-eval-and-data` (pre-change snapshot preserved on branch `snapshot/pre-clean-eval`)
* **Checkpoint**: `checkpoints/exp01_clean_lags32/best_policy.pt`
* **Benchmark JSON**: `checkpoints/exp01_clean_lags32/eval_benchmark_sim_clean.json`
* **Diagnostic GIFs**: `checkpoints/exp01_clean_lags32/gifs/` (`demo_0000_task0.gif`, `demo_0101_task1.gif`, `demo_0200_task2.gif`, `test_t0_s9013.gif`)
* **Dataset**: `data/sim_demos.h5` (`300` expert trajectories: `100` per task, `50` `sim_clean` + `50` `sim_dr`, `144` steps per episode at `20 Hz`)

### 1. What Changed vs. Initial Prototype (`snapshot/pre-clean-eval`)

1. **Restored Grasp, Release, and Retract Transitions (`training/dataset.py`)**:
   - Removed aggressive frame dropping (`clamp_dwell`, `release_lag`, `retract_bridge`, and terminal truncation) inside `filter_episode_frames` so that the policy trains on the complete `8`-segment trajectory (`approach -> descend -> clamp -> lift -> transit -> lower -> release -> retract`).
   - Retained `43,131` valid training transitions across the `300` episodes (`337` batches per epoch at `batch_size = 128`).
2. **Removed All Time-Index Overrides & Open-Loop Hacks (`training/dataset.py`, `training/model.py`, `evaluation/evaluator.py`)**:
   - Removed the `step < 16` gripper `0.68` override in both dataset preparation and closed-loop evaluation.
   - Removed `home_anchor_prob`, `pan_noise_std`, and `dq[:, 0] = 0` masking.
   - Replaced single-step delta history with causal multi-scale proprioception history `proprio_history_lags = (0, 4, 8, 16, 32)` (`0.0 s, 0.2 s, 0.4 s, 0.8 s, 1.6 s`), producing a `30D` history vector (`6D` current posture + `4 x 6D` multi-scale displacements).
3. **Evaluation Horizon & First-Pass Success (`evaluation/evaluator.py`)**:
   - Increased default rollout horizon from `154` steps to `max_steps = 280` (`14.0 s` at `20 Hz`, giving `136` steps of slack over the `144`-step expert trajectory).
   - Recorded episode success the first time `env.check_constraints(task_id)["all_constraints_passed"]` holds during closed-loop execution.
   - Selected `best_policy.pt` using deterministic ODE action-chunk MSE (`val_ode_mse`) evaluated in `policy.eval()` mode across `512` validation samples.

### 2. Training Summary (`20` Epochs, `sim_only`)

| Metric | Epoch 01 | Epoch 05 | Epoch 10 | Epoch 15 | Epoch 20 (Best) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flow Matching Loss (`loss`)** | `0.71162` | `0.14677` | `0.09883` | `0.08083` | **`0.07355`** |
| **Arm Velocity MSE (`arm`)** | `0.78116` | `0.16055` | `0.10793` | `0.08907` | **`0.08168`** |
| **Gripper Velocity MSE (`grip`)** | `0.57255` | `0.11921` | `0.08062` | `0.06436` | **`0.05729`** |
| **Eval-Mode ODE Chunk MSE (`val_ode_mse`)** | `0.02878` | `0.00440` | `0.00240` | `0.00167` | **`0.00140`** |

### 3. Benchmark Evaluation (`120` Episodes Total: `60` Train Split + `60` Held-Out Test Split)

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

### 4. Key Takeaways from Experiment 01

1. **Second Half of Trajectory (Carry, Release, and Retract) Works End-to-End**:
   - On `test_t0_s9013` (`Task 0`, held-out seed `9013`), where the initial reach landed within `0.62 cm` of the pen holder, the policy grasped the pen holder (`0.09 CLAMPED`), lifted it to `13.5 cm`, carried it into the bowl (`Src->Tgt XY = 1.65 cm`), opened the gripper to `0.74 (OPEN)`, retracted cleanly (`Pinch->Src XY = 5.2 cm`), and passed strict `all_constraints_passed` at `Step 149/280`.
   - Even on failed grasp rollouts like `demo_0101` (`min_src_xy = 1.13 cm`), after clamping empty air beside the cup at `Step 056`, the policy still lifted, carried to the bowl (`Pan q0 = +0.19 rad` vs `GT = +0.21 rad`), opened the gripper to `0.72 (OPEN)`, and retracted.
2. **Primary Bottleneck Identified (The Copycat / Proprioceptive Shortcut Problem)**:
   - Counterfactual testing on `exp01_clean_lags32` between a far-left Task 0 demo (`demo_0001`, `GT_left = -0.88 rad`) and a far-right Task 0 demo (`demo_0005`, `GT_right = +0.74 rad`) revealed that the policy is almost completely ignoring the cameras at `Step 00..15` and instead extrapolating `proprio_history`:

| Timestep | True Target Pan (`GT_left` vs `GT_right`) | Original `(Left Cam, Left Proprio)` | **Camera Swapped** `(Right Cam, Left Proprio)` | **Proprio Swapped** `(Left Cam, Right Proprio)` | Camera Sensitivity (`swap_cam_diff`) | Proprio Sensitivity (`swap_prop_diff`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Step 00`** (at `q_home`, `dq = 0`) | `-0.700` vs `+0.587 rad` | **`+0.300 rad`** (wrong sign) | **`+0.354 rad`** | `+0.300 rad` | **`0.0126 rad` (blind)** | `0.0000 rad` |
| **`Step 05`** (`0.25 s` into reach) | `-0.866` vs `+0.726 rad` | `-0.722 rad` | **`-0.570 rad`** (still goes LEFT) | **`+0.633 rad`** (flips to RIGHT) | **`0.0331 rad`** | **`0.1801 rad` (5.4x larger)** |
| **`Step 15`** (`0.75 s` into reach) | `-0.881` vs `+0.742 rad` | `-0.813 rad` | **`-0.348 rad`** | **`+0.270 rad`** | `0.1074 rad` | **`0.1964 rad`** |

   - Why this happens: in `data/sim_demos.h5`, `99.3%` of steps (`143/144`) have non-zero velocity deltas `dq = q(t) - q(t-k)` and lie on the deterministic ray $q(t) = q_{\text{home}} + s(t)(q_{\text{hover\_src}} - q_{\text{home}})$. The MLP achieves `val_ode_mse = 0.00140` by extrapolating `proprio_history` without training the vision encoders.
   - **Next Experiment (`exp02`)**: Break the copycat shortcut by removing multi-step velocity history (`proprio_history_lags = (0,)`), increasing proprioception dropout/noise (`proprio_drop_prob = 0.50`, `proprio_noise_std = 0.08`), and enforcing direct visual target supervision from the 3 cameras.


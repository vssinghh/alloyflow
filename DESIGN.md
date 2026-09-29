# AlloyFlow: System Design

**AlloyFlow** is a multi-task robot learning project for the **SO-ARM101** robot arm (`5` arm joints + `1` gripper = `6` joints total). It trains a single neural network policy to perform 3 tabletop tasks and compares how well the robot learns from simulation data, real-world data, finetuning, and co-training.

***

## 1. Core Objectives

### Objective 1: Train One Model to Perform 3 Tasks
Train **one single model** that takes 3 camera images (`128x128`), the robot's `6` current joint positions, and a task ID (`0`, `1`, or `2`), and outputs the robot's next `16` steps of joint movements.

All 3 tasks use the same tabletop setup with **4 everyday objects**:
1. **Pen Holder**: A small desktop pen holder without a pen (`4 to 5 cm` wide).
2. **Small Cup**: A lightweight plastic, paper, or espresso cup.
3. **Bowl**: A target container (`~10 cm` wide).
4. **Rubik's Cube**: A standard `5.6 cm` cube used as a flat base for stacking.

By changing only the task ID (`0`, `1`, or `2`), the same model performs 3 different tasks:
1. **Task 0 (`pick_pen_holder_to_bowl`)**: Pick up the pen holder and place it inside the bowl.
2. **Task 1 (`pick_cup_to_bowl`)**: Ignore the pen holder, pick up the small cup, and place it inside the bowl. (Tests picking a different object for the same destination).
3. **Task 2 (`stack_cup_on_cube`)**: Pick up the small cup and stack it on top of the Rubik's cube. (Tests picking the same object as Task 1, but placing it on a different target).

### Objective 2: Compare 4 Training Modes
Using the exact same model architecture and test episodes, compare how the robot performs across **4 training modes**:
1. **Mode 1: `Sim-Only` (`100%` Simulation Data)**
   - **Training**: Trained only on MuJoCo simulation demos (`100` demos per task, `300` total).
   - **Goal**: Measure baseline performance in simulation, and test how well a simulation-only model transfers directly to the real robot without any real training data.
2. **Mode 2: `Real-Only` (`100%` Real Robot Data)**
   - **Training**: Trained only on a small set of real robot demos (`20` demos per task, `60` total).
   - **Goal**: Measure how well the real robot learns from limited real data, and where it fails when objects are placed in new positions.
3. **Mode 3: `Sim -> Real Finetune` (Pretrain in Sim, then Finetune on Real)**
   - **Training**: Start from the `Mode 1 (Sim-Only)` model weights, then continue training on the `20` real demos per task (`60` total).
   - **Goal**: Measure if pretraining in simulation helps real-world performance, and check if finetuning causes the model to forget how to solve the simulation tasks.
4. **Mode 4: `Sim + Real Co-Training` (`50%` Sim + `50%` Real in Every Batch)**
   - **Training**: Train from scratch where every batch of `128` samples pulls `64` samples from the `300` simulation demos and `64` samples from the `60` real robot demos.
   - **Goal**: Test if mixing simulation and real data in every batch gives the best real-world generalization while keeping high performance in simulation.

***

## 2. Robot Hardware & Simulation Setup (`SO-ARM101`)

* **Robot Arm**: 6-DoF **SO-ARM101** (`5` Feetech STS3215 arm servos: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` + `1` gripper servo).
* **Inputs & Outputs (`20 Hz`)**:
  - **Input**: `6` current joint positions + `3` RGB camera images (`128x128`) + `1` task ID (`0, 1, 2`).
  - **Output**: A chunk of `16` future joint position targets (`16 x 6`).
* **Cameras**: 3 synchronized `128x128` RGB views (`third_person_cam`, `overhead_cam`, `wrist_cam`).

***

## 3. Repository Structure & CLI Commands

```text
alloyflow/                        # Repository Root
├── assets/
│   └── so101/                    # SO-ARM101 MuJoCo XML + 4 tabletop objects
│
├── envs/                         # Sim and Real environments (shared reset/step API)
│   ├── __init__.py
│   ├── base.py                   # Shared SO101Env interface & 3 Task IDs
│   ├── sim_env.py                # MuJoCo simulation environment
│   ├── real_env.py               # Physical SO-ARM101 robot environment
│   └── tests/
│
├── collection/                   # Step 1: Collect demos (Sim or Real -> .h5)
│   ├── __init__.py
│   ├── sim_expert.py             # Scripted IK planner for Sim demos
│   ├── real_teleop.py            # Leader-follower recorder for Real demos
│   ├── collector.py              # Shared HDF5 file writer
│   ├── __main__.py               # CLI: python -m collection
│   └── tests/
│
├── training/                     # Step 2: Model & 4-mode training loop
│   ├── __init__.py
│   ├── config.py                 # Training hyperparameters
│   ├── model.py                  # Vision Flow Matching policy network
│   ├── flow_matching.py          # Flow matching loss & ODE sampler
│   ├── dataset.py                # HDF5 loader & Sim+Real batch mixer
│   ├── trainer.py                # Modes: sim_only, real_only, finetune, cotrain
│   ├── __main__.py               # CLI: python -m training
│   └── tests/
│
├── evaluation/                   # Step 3: Evaluate on Sim, Real, or Both
│   ├── __init__.py
│   ├── evaluator.py              # Rollout runner & success rate tracker
│   ├── visualizer.py             # Camera HUD overlay & video recorder
│   ├── __main__.py               # CLI: python -m evaluation
│   └── tests/
│
├── DESIGN.md
├── EXPERIMENT_LOG.md
└── pyproject.toml
```

### CLI Commands
1. **Collect Demos (`Sim` or `Real`)**:
   ```bash
   uv run python -m collection --domain sim --task all --episodes 100
   uv run python -m collection --domain real --task 0 --episodes 20
   ```
2. **Train Policy (`sim_only`, `real_only`, `finetune`, `cotrain`)**:
   ```bash
   uv run python -m training --mode cotrain --real-ratio 0.5
   ```
3. **Evaluate Policy (`sim`, `real`, or `both`)**:
   ```bash
   uv run python -m evaluation --checkpoint checkpoints/cotrain/best_policy.pt --domain both
   ```
4. **Run Tests**:
   ```bash
   uv run pytest
   ```

***

## 4. Data Collection (`collection/`)

Both simulation and real-world collection save data in the exact same `.h5` file format at **`20 Hz` (one step every `50 ms`)** using identical joint units:
- **Arm Joints (`0..4`)**: Radians (`float32`), measured relative to the robot's home pose.
- **Gripper (`5`)**: Normalized from `0.0` (closed) to `1.0` (open).
- **Camera Images**: Three `128x128` RGB images (`uint8`) per step.

### 4.1 Simulation Data Collection (`--domain sim`)
`collection/sim_expert.py` solves each task in MuJoCo using 6-DoF Inverse Kinematics (IK) at 5 key poses (`Hover`, `Descend`, `Grasp & Lift`, `Move to Goal`, `Place & Release`) and interpolates smoothly in `6D` joint space so the arm follows a clean upward arc over tabletop clutter.

* **Dataset Scale & Two-Step Randomization Plan (`100` demos per task, `300` total)**:
  1. **Object `(X, Y)` Positions**: Always randomized across the full reachable tabletop workspace on every episode.
  2. **Visual & Physical Domain Randomization (`--dr`)**:
     - First, we verify the planner and policy on **Clean Simulation** (`50` demos per task with fixed studio lighting and camera mounts) to confirm `90%+` task success.
     - Next, we collect **`50` Domain-Randomized (`DR`) demos per task** (varying lighting direction, object/tabletop colors, $\pm 1\text{ cm}$ camera mount shifts, and object mass `0.7x to 1.5x`), giving **`100` Sim demos per task (`300` total)** in `data/alloy_sim_demos.h5`.

* **Mandatory Quality & Constraint Checks (Verified Before Saving Any Demo)**:
  Every simulated episode must pass 4 automated checks before it is written to `.h5` (failed seeds are skipped automatically):
  1. **Spawn Clearance Check (Before Step 0)**: All 4 objects (`pen_holder`, `cup`, `bowl`, `rubiks_cube`) must spawn inside the reachable workspace with at least **`8 cm` center-to-center distance** between every pair of objects and zero initial collisions.
  2. **Continuous Motion Check (No Stationary Dwell Frames)**: The planner never pauses in place during grasp or release, and any step where both the arm and gripper barely move ($\|\Delta q_t\| < 10^{-3}$) is trimmed so the policy never learns to freeze mid-episode.
  3. **Bystander Object Check (Zero Unwanted Collisions)**: Non-target objects on the table must not be tipped over or pushed more than **`1.5 cm`** from their starting positions during the run.
  4. **Task Completion & Settling Check (End of Episode)**: After releasing the object and retracting the gripper upward, the simulation steps forward `10` extra settling frames and verifies:
     - **Task 0 (`pick_pen_holder_to_bowl`)**: `pen_holder` is upright inside the `bowl` radius.
     - **Task 1 (`pick_cup_to_bowl`)**: `cup` is upright inside the `bowl` radius.
     - **Task 2 (`stack_cup_on_cube`)**: `cup` is upright, centered within `2.0 cm` of the `rubiks_cube`, and resting stably on its top face with the gripper clear.

### 4.2 Real Robot Data Collection (`--domain real`)
`collection/real_teleop.py` records demonstrations on the physical SO-ARM101 using **leader-follower teleoperation** at `20 Hz`:
1. **Leader Arm (Motor Torque Off)**: You move the leader arm by hand. Every `50 ms`, the script reads its `6` servo positions over USB.
2. **Follower Arm (Motor Torque On)**: The follower arm copies the leader arm's `6` joint positions in real time while the script records the follower's actual joint positions, the commanded joint targets, and the 3 USB camera frames.
3. **Keyboard Controls**:
   - Press `SPACE` to start recording an episode, and `SPACE` again when finished to save it.
   - Press `r` (or `BACKSPACE`) if you make a mistake to throw away the current attempt and try again.

We collect **`20` clean demos per task (`60` total)** into `data/alloy_real_demos.h5`.

### 4.3 HDF5 File Format (`data/alloy_sim_demos.h5` & `data/alloy_real_demos.h5`)
```text
alloy_<domain>_demos.h5
├── attrs:
│   ├── domain: "sim" | "real"
│   ├── control_hz: 20
│   └── cameras: ["third_person_cam", "overhead_cam", "wrist_cam"]
└── data/
    └── demo_0/
        ├── attrs:
        │   ├── task_id: 0 | 1 | 2
        │   ├── domain: "sim" | "real"
        │   ├── seed: int
        │   └── num_samples: T
        ├── actions                    # float32 (T, 6) (commanded joint targets)
        └── obs/
            ├── proprio                # float32 (T, 6) (current joint positions)
            ├── rgb_third_person_cam   # uint8   (T, 128, 128, 3)
            ├── rgb_overhead_cam       # uint8   (T, 128, 128, 3)
            └── rgb_wrist_cam          # uint8   (T, 128, 128, 3)
```

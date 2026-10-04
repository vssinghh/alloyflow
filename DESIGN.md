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
│   ├── stage1_pretrain.png       # Phase 1 of 3: Stage 1 Vision Pretraining diagram
│   ├── stage2_training.png       # Phase 2 of 3: Stage 2 Flow Matching Training diagram
│   ├── stage3_inference.png      # Phase 3 of 3: Closed-Loop 20 Hz Inference diagram
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
├── training/                     # Step 2: Model, Stage 1 pretraining & 4-mode training loop
│   ├── __init__.py
│   ├── config.py                 # Training hyperparameters
│   ├── model.py                  # Vision Flow Matching policy + 4 parallel side-branch pos heads
│   ├── pretrain_vision.py        # Stage 1 3,000-scene layout generator & vision pretrainer
│   ├── flow_matching.py          # Flow matching loss, co-supervised pos loss & ODE sampler
│   ├── dataset.py                # HDF5 loader, 12D object XY labels & Sim+Real batch mixer
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
├── README.md
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
3. **Evaluate Policy (`120`-Episode Benchmark or Per-Demo Diagnostic GIFs)**:
   ```bash
   uv run python -m evaluation --checkpoint checkpoints/sim_only/best_policy.pt --benchmark --episodes 20
   uv run python -m evaluation --checkpoint checkpoints/sim_only/best_policy.pt --demos demo_0000,demo_0101,demo_0200
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
     - Next, we collect **`50` Domain-Randomized (`DR`) demos per task** (varying lighting direction, object/tabletop colors, $\pm 1\text{ cm}$ camera mount shifts, and object mass `0.7x to 1.5x`), giving **`100` Sim demos per task (`300` total)** in `data/sim_demos.h5`.

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

We collect **`20` clean demos per task (`60` total)** into `data/real_demos.h5`.

### 4.3 HDF5 File Format (`data/sim_demos.h5` & `data/real_demos.h5`)
```text
<domain>_demos.h5
├── attrs:
│   ├── num_episodes: N
│   ├── num_cameras: 3
│   ├── control_hz: 20
│   └── camera_names: ["third_person_cam", "overhead_cam", "wrist_cam"]
└── demo_0000/
    ├── attrs:
    │   ├── task_id: 0 | 1 | 2
    │   ├── task_name: str
    │   ├── domain: "sim_clean" | "sim_dr" | "real"
    │   ├── seed: int
    │   └── num_steps: T
    ├── actions                        # float32 (T, 6) (commanded joint targets)
    └── obs/
        ├── proprio                    # float32 (T, 6) (current joint positions)
        ├── rgb_third_person_cam       # uint8   (T, 128, 128, 3)
        ├── rgb_overhead_cam           # uint8   (T, 128, 128, 3)
        └── rgb_wrist_cam              # uint8   (T, 128, 128, 3)
```

***

## 5. Policy Architecture & 3-Phase Training/Inference Pipeline (`training/`)

All 4 training modes (`sim_only`, `real_only`, `finetune`, `cotrain`) train the same **`TaskConditionedVisionFlowPolicy`** (`AlloyFlowPolicy`) network. To prevent the vision encoders from underfitting on `300` demonstrations, the pipeline operates across **three distinct phases**:
1. **Phase 1 (`Stage 1: Vision Pretraining`)**: Strictly vision-only (`encode_vision_tokens`). Pretrains the 3 camera CNNs, task embedding, and 3 parallel per-camera `12D` object position heads (`aux_cam_pos_heads`) on `3,000` labeled tabletop scenes (`data/loc_layouts_3000.npz`).
2. **Phase 2 (`Stage 2: Flow Matching Training + Co-Supervised Aux Position Loss`)**: Trains the full policy end-to-end on trajectory demonstrations while keeping the 3 parallel per-camera position heads active as training-only side branches (`L_total = L_CFM + 0.5 * L_pos`). The three `32D` camera tokens (`[tok_third_person, tok_overhead, tok_wrist] = 96D`) are concatenated directly with `z_prop (64D)` and `z_task (32D)` into `z_fused (192D)` with zero attention bottleneck.
3. **Phase 3 (`Stage 3: Closed-Loop 20 Hz Inference`)**: Ignores all 3 position heads (`1,026,435` active parameters), computes the `192D` observation vector `z_fused` once per control step, and integrates the `4-Block Flow ResMLP` over `10` Euler ODE steps.

### 5.1 Three-Phase Pipeline Diagrams

#### Phase 1 of 3: Stage 1 Vision Pretraining (`3,000` Steps on `data/loc_layouts_3000.npz`)

![Phase 1 of 3: Stage 1 Vision Pretraining](assets/stage1_pretrain.png)

* **Dataset (`data/loc_layouts_3000.npz`)**: `3,000` static 3-camera tabletop scenes (`50%` clean + `50%` domain-randomized, `50%` at home pose + `50%` at random demo arm postures) labeled with `12D` tabletop `(X, Y)` coordinates:
  $$\mathbf{y}_{\text{pos}} = [x_{\text{src}}, y_{\text{src}}, \, x_{\text{tgt}}, y_{\text{tgt}}, \, x_{\text{pen}}, y_{\text{pen}}, \, x_{\text{cup}}, y_{\text{cup}}, \, x_{\text{bowl}}, y_{\text{bowl}}, \, x_{\text{cube}}, y_{\text{cube}}] \in \mathbb{R}^{12}$$
* **Strictly Vision-Only (`encode_vision_tokens`)**:
  - **`[TRAINED]`**: `3x SpatialSoftmaxCNN`, `Task Embedding (3 -> 32D)`, and the **3 parallel per-camera `12D` position heads** (`aux_cam_pos_heads`: one each for `third_person_cam`, `overhead_cam`, and `wrist_cam`).
  - **`Unused`**: `ProprioMLP (6D -> 64D)` and the `4-Block Flow ResMLP` are not called or touched in Stage 1.
* **Stage 1 Loss**:
  $$\mathcal{L}_{\text{Stage 1}} = 1.0 \cdot \text{MSE}_{\text{overhead}} + 1.0 \cdot \text{MSE}_{\text{third\_person}} + 0.5 \cdot \text{MSE}_{\text{wrist}}$$

#### Phase 2 of 3: Stage 2 Flow Matching Training + Co-Supervised Aux Position Loss (`20` Epochs)

![Phase 2 of 3: Stage 2 Flow Matching Training](assets/stage2_training.png)

* **All modules `[TRAINED]`**: Starts from the Stage 1 vision weights and trains the entire network (`camera_encoders`, `proprio_mlp`, `task_embedding`, `4-Block Flow ResMLP`, and the 3 parallel per-camera position heads) for `20` epochs (`batch_size = 128`, `lr = 5e-4`, `1,056,039` trainable parameters).
* **Direct Camera Concatenation & Training-Only Parallel Side Branches (Never in `z_fused`)**:
  - The 3 per-camera position heads (`aux_cam_pos_heads`) run **in parallel** (none feeds into another) and sit strictly in a training-only side branch (`predict_aux_positions`).
  - Their `12D` coordinate predictions **never** enter `z_fused`. Instead, the three `32D` camera tokens are concatenated directly in fixed slot order (`CAMERA_NAMES`) with `z_prop` and `z_task` to form `z_fused`:
    $$\mathbf{z}_{\text{fused}} = [\text{tok}_{\text{third\_person}}(32\text{D}) \,;\, \text{tok}_{\text{overhead}}(32\text{D}) \,;\, \text{tok}_{\text{wrist}}(32\text{D}) \,;\, \mathbf{z}_{\text{prop}}(64\text{D}) \,;\, \mathbf{z}_{\text{task}}(32\text{D})] \in \mathbb{R}^{192}$$
  - During training (`model.train()`), `wrist_cam_drop_prob = 0.05` zeros out the wrist camera token slice `[64:96]` in `z_fused` on `5%` of samples so the policy never over-relies on a single occluded view.
* **Stage 2 Total Loss (`compute_loss` in `training/flow_matching.py`)**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + 0.5 \cdot \mathcal{L}_{\text{pos}}, \qquad \mathcal{L}_{\text{pos}} = 0.5 \cdot \mathcal{L}_{\text{pos, demo}} + 0.5 \cdot \mathcal{L}_{\text{pos, replay}}$$
  - $\mathcal{L}_{\text{pos, demo}}$: Computed on the `128` demo frames in the batch (`src_xy` and the picked object's `(X, Y)` coordinates are masked out via `obj_xy_mask` for `raw_idx >= 45` once the object is lifted off the table).
  - $\mathcal{L}_{\text{pos, replay}}$: Computed via `encode_vision_tokens` on `32` static layouts replayed from `data/loc_layouts_3000.npz` on every batch so the CNNs never forget wide-workspace localization.
  - Both parts average the 3 parallel per-camera head MSEs with weights `1.0` (`overhead_cam`), `1.0` (`third_person_cam`), and `0.25` (`wrist_cam`) (`w_total = 2.25`).

#### Phase 3 of 3: Closed-Loop Inference on the Robot (`20 Hz`)

![Phase 3 of 3: Closed-Loop Inference at 20 Hz](assets/stage3_inference.png)

* **Zero inference overhead**: During `predict_action_chunk`, all 3 position heads (`29,604` parameters) are completely ignored (`1,026,435` active inference parameters). The policy calls `encode_observations` once per `50 ms` control step to build $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{192}$, integrates `Flow ResMLP` over `10` Euler ODE steps ($\Delta \tau = 0.1$), and blends overlapping `16 x 6` chunks via Temporal Ensembling ($w_i = \exp(-0.05 \cdot i)$).

#### Component Summary Across All 3 Phases

| Component | Stage 1: Vision Pretraining | Stage 2: Policy Training | Stage 3: `20 Hz` Inference |
| :--- | :--- | :--- | :--- |
| **`3x Spatial Softmax CNN`** | **`[TRAINED]`** from scratch | **`[TRAINED]`** (fine-tuned) | **Used** (frozen weights) |
| **`Task Embedding (3 -> 32D)`** | **`[TRAINED]`** from scratch | **`[TRAINED]`** (fine-tuned) | **Used** (frozen weights) |
| **`Proprio MLP (6 -> 64D)`** | **Unused** | **`[TRAINED]`** from scratch | **Used** (frozen weights) |
| **`3 Per-Camera Pos Heads`** | **`[TRAINED]`** (`w = 1.0, 1.0, 0.5`) | **`[TRAINED]`** (`w = 1.0, 1.0, 0.25`) | **Ignored** (not called) |
| **`z_fused (192D)`** | Not built (only `3x 32D` `cam_tokens` built) | **Built** (`[tok_tp, tok_ov, tok_wr, z_prop, z_task]`) | **Built once per step** (`192D`) |
| **`4-Block Flow ResMLP`** | **Unused** | **`[TRAINED]`** from scratch | **Used** (`10` Euler ODE steps) |

### 5.2 Detailed Layer-by-Layer Specification

1. **Task Embedding (`nn.Embedding(3, 32)`)**:
   - Maps discrete `task_id in {0, 1, 2}` to a learnable `32D` task vector $\mathbf{z}_{\text{task}} \in \mathbb{R}^{32}$.
   - Enters **three** places across the architecture: (a) Task `FiLM` inside each of the 3 camera CNNs, (b) the `192D` fused vector $\mathbf{z}_{\text{fused}}$ entering the `ResMLP`, and (c) the input of all 3 parallel per-camera auxiliary position heads during training.

2. **Proprioception & Causal History MLP (`proprio_input_dim -> 64D -> 64D`)**:
   - Normalizes raw `6D` joint state $\mathbf{q}_t$ (`5` arm joints in radians + `1` gripper in `[0, 1]`) via dataset z-score statistics ($\tilde{\mathbf{q}}_t = (\mathbf{q}_t - \boldsymbol{\mu}_q) / \boldsymbol{\sigma}_q$, stored inside checkpoint buffers).
   - Configured via `proprio_history_lags` (default single-frame `(0,)` = `6D`, removing multi-step velocity copycat shortcuts).
   - During Stage 2 training (`model.train()`), applies Gaussian jitter (`proprio_noise_std = 0.02`) to the normalized joint inputs and modality dropout (`proprio_drop_prob = 0.10`) on the resulting `64D` embedding $\mathbf{z}_{\text{prop}} \in \mathbb{R}^{64}$.
   - Passes the input through two `Linear + LayerNorm(64) + SiLU` layers to produce $\mathbf{z}_{\text{prop}} \in \mathbb{R}^{64}$.

3. **Tri-Camera Spatial Softmax CNN with Task `FiLM` (`3 x SpatialSoftmaxConvNet`)**:
   - Three independent encoders (one each for `third_person_cam`, `overhead_cam`, and `wrist_cam`).
   - During training, applies $\pm 4\text{ px}$ random bilinear shift augmentation (`shift_pad = 4`), keypoint coordinate noise (`keypoint_noise = 0.01`), and stochastic wrist camera token dropout (`wrist_cam_drop_prob = 0.05`).
   - **4 Conv Blocks (`GroupNorm + SiLU` down to `16x16` Spatial Grid)**:
      - `Conv1`: `(3 -> 32, k=5, stride=2, pad=2) + GroupNorm(4, 32) + SiLU` $\to$ `(B, 32, 64, 64)`
      - `Conv2`: `(32 -> 64, k=3, stride=2, pad=1) + GroupNorm(8, 64) + SiLU` $\to$ `(B, 64, 32, 32)`
      - `Conv3`: `(64 -> 64, k=3, stride=2, pad=1) + GroupNorm(8, 64) + SiLU` $\to$ `(B, 64, 16, 16)`
      - `Conv4`: `(64 -> 32, k=3, stride=1, pad=1) + GroupNorm(4, 32)` + **Task `FiLM`** ($\tilde{f}_c = (1 + \gamma_c(\mathbf{z}_{\text{task}})) f_c + \beta_c(\mathbf{z}_{\text{task}})$) $\to$ `(B, 32, 16, 16)` (no pre-softmax `SiLU` so negative background logits are suppressed cleanly)
   - **2D Spatial Softmax (`16x16` Grid, `32` Keypoints, Learnable Temperature)**:
      - Softmaxes each `16x16` (`256`-cell) heatmap with a sharp learnable temperature parameter (`init_temperature = 0.1`) into a 2D probability distribution $P_c(u, v)$ and computes the expected coordinate $(\mu_{c,x}, \mu_{c,y}) \in [-1, +1]^2$ across all `32` channels (`64D` coordinate vector per camera).
   - **Camera Token Projection**:
      - `Linear(64 -> 32) + LayerNorm(32) + SiLU` outputs $\mathbf{z}_{\text{tp}}, \mathbf{z}_{\text{ov}}, \mathbf{z}_{\text{wr}} \in \mathbb{R}^{32}$.

4. **Direct Multi-Camera Concatenation (`3 x 32D = 96D`) & Fusion (`192D`)**:
   - Directly concatenates the three camera tokens in fixed slot order (`[0:32]` = `third_person_cam`, `[32:64]` = `overhead_cam`, `[64:96]` = `wrist_cam`) without cross-camera attention mixing:
     $$\mathbf{z}_{\text{cam}} = [\mathbf{z}_{\text{tp}} (32\text{D}) \,;\, \mathbf{z}_{\text{ov}} (32\text{D}) \,;\, \mathbf{z}_{\text{wr}} (32\text{D})] \in \mathbb{R}^{96}$$
   - During training (`model.train()`), applies stochastic wrist camera token dropout (`wrist_cam_drop_prob = 0.05`) on slice `[64:96]`.
   - **Fused Conditioning Vector (`192D`)**:
     $$\mathbf{z}_{\text{fused}} = [\mathbf{z}_{\text{tp}} (32\text{D}) \,;\, \mathbf{z}_{\text{ov}} (32\text{D}) \,;\, \mathbf{z}_{\text{wr}} (32\text{D}) \,;\, \mathbf{z}_{\text{prop}} (64\text{D}) \,;\, \mathbf{z}_{\text{task}} (32\text{D})] \in \mathbb{R}^{192}$$

5. **Three Parallel Training-Only Auxiliary Position Heads (`aux_cam_pos_heads`)**:
   - Three independent MLPs (`Linear(64 -> 128) + SiLU + Linear(128 -> 12)`), one per camera, each taking `[tok_c(32D), z_task(32D)]` and predicting normalized `12D` object `(X, Y)` coordinates (`obj_xy_mean` and `obj_xy_std` stored as model buffers).

6. **Optimal-Transport Conditional Flow Matching Head (`4-Block ResMLP` + `K = 4` Stratified Sampling)**:
   - **Target Action Chunk ($x_1$)**: Next `16` steps of `6D` joint commands `(16, 6)` (`0.8 s` at `20 Hz`), z-score normalized using `action_mean` and `action_std`. At episode boundaries (`t > T - 16`), steps beyond `T - 1` repeat `actions[T - 1]` so the robot holds its final retracted pose cleanly.
   - **Nonlinear Observation Conditioning (`obs_proj`)**: Projects $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{192}$ through a 2-layer nonlinear MLP (`Linear(192 -> 256) + LayerNorm(256) + SiLU + Linear(256 -> 256) + Dropout`) before summing with `act_proj` and `SinusoidalTimeEmbedding` ($\tau \in [0, 1]$).
   - **Stratified Flow Amortization (`K = 4`)**:
     - Computes $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{192}$ **once** per minibatch through the 3 CNNs and evaluates the lightweight `ResMLP` across `K = 4` stratified flow timesteps $\tau_{i,k} = \frac{k + u_{i,k}}{4}$ for $k \in \{0, 1, 2, 3\}$ and $u_{i,k} \sim \mathcal{U}(0, 1)$.
   - **Straight-Line Interpolation, Velocity Loss & Validation Selection**:
     - Samples $x_0 \sim \mathcal{N}(0, I)$, forms $x_\tau = (1 - \tau) x_0 + \tau x_1$, and trains $v_\theta(x_\tau, \tau \mid \mathbf{z}_{\text{fused}})$ to match $u_\tau = x_1 - x_0$ using dimension-weighted MSE with `gripper_weight = 2.5` on joint `5`.
     - At the end of each epoch, evaluates deterministic 10-step Euler ODE action-chunk MSE (`val_ode_mse`) in `policy.eval()` mode on `val_samples_per_epoch = 512` transitions and saves the lowest-error model to `best_policy.pt`.
   - **Closed-Loop Inference (`20 Hz`)**:
     - Integrates $v_\theta$ from $\tau = 0 \to 1$ in `10` Euler steps ($\Delta \tau = 0.1$) and blends overlapping `16`-step predictions using **Temporal Ensembling** ($w_i = \exp(-0.05 \cdot i)$).

### 5.3 The 4 Training Modes & Default Hyperparameters (`training/config.py`)

| Mode Flag | Training Dataset(s) | Batch Sampling | Learning Rate | Normalization Stats (`norm_stats`) |
| :--- | :--- | :--- | :---: | :--- |
| **`sim_only`** | `data/sim_demos.h5` (`300` Sim demos) | `128` Sim samples/batch | `5e-4` | Computed from `sim_demos.h5` & saved in checkpoint |
| **`real_only`** | `data/real_demos.h5` (`60` Real demos) | `128` Real samples/batch | `5e-4` | Computed from `real_demos.h5` & saved in checkpoint |
| **`finetune`** | Pretrained `sim_only` $\to$ `data/real_demos.h5` | `128` Real samples/batch | `1e-4` | **Locked from `sim_only` checkpoint** (no stat drift) |
| **`cotrain`** | `data/sim_demos.h5` + `data/real_demos.h5` | **`64` Sim + `64` Real** (`real_ratio=0.5`) | `5e-4` | Computed across combined training set & saved |

| Hyperparameter | Default Value | Purpose |
| :--- | :---: | :--- |
| `chunk_size` (`H`) | `16` | `0.8 s` future action horizon at `20 Hz`. |
| `proprio_history_lags` | `(0,)` | Single-frame `6D` proprioception (`(0, 4, 8, ...)` optional for multi-step history). |
| `num_keypoints` / `vision_feat_dim` | `32` / `32` | `32` `(x, y)` keypoints (`64D` from `16x16` grid, `temp=0.1`) projected to `32D` per camera. |
| `task_emb_dim` / `proprio_emb_dim` | `32` / `64` | Task lookup embedding and 2-layer proprioception MLP dimensions. |
| `fused_dim` | `192` | `96D (3 concatenated cameras) + 64D (proprio) + 32D (task)`. |
| `hidden_dim` / `num_res_blocks` | `256` / `4` | `ResMLP` width and depth (`1,056,039` trainable params; `1,026,435` active at inference). |
| `pretrain_loc_steps` / `loc_data_path` | `3000` / `data/loc_layouts_3000.npz` | Stage 1 vision pretraining steps and rendered `3,000`-scene layout cache. |
| `aux_pos_loss_weight` | `0.5` | Stage 2 co-supervision weight on $\mathcal{L}_{\text{pos}} = 0.5 \cdot \mathcal{L}_{\text{pos, demo}} + 0.5 \cdot \mathcal{L}_{\text{pos, replay}}$. |
| `num_flow_samples` (`K`) | `4` | Stratified flow timesteps per CNN forward pass. |
| `gripper_weight` | `2.5` | Loss weight on gripper dimension `5` (`1.0` on arm dimensions `0..4`). |
| `shift_pad` / `keypoint_noise` | `4` / `0.01` | Random $\pm 4\text{ px}$ image shift and `0.01` training keypoint jitter. |
| `proprio_noise_std` / `proprio_drop_prob` | `0.02` / `0.10` | Training Gaussian jitter on normalized $\tilde{\mathbf{q}}$ and `z_prop` dropout probability. |
| `wrist_cam_drop_prob` / `dropout` | `0.05` / `0.05` | Wrist camera token dropout and `ResMLP` activation dropout. |
| `batch_size` / `epochs` | `128` / `20` | `80` effective flow passes per sample (`20 * K`), AdamW (`wd = 1e-4`). |
| `val_samples_per_epoch` | `512` | Validation samples used to compute eval-mode `val_ode_mse` for `best_policy.pt`. |
| `ode_steps` / `temporal_ensemble_decay` | `10` / `0.05` | Euler ODE integration steps and $w_i = \exp(-0.05 \cdot i)$ blending. |

***

## 6. Closed-Loop Evaluation & Visual Diagnostics (`evaluation/`)

`evaluation/evaluator.py` (`SimPolicyEvaluator`) and `evaluation/visualizer.py` (`RolloutVisualizer`) execute closed-loop physics rollouts in `SimEnv` at `20 Hz` (`max_steps = 280`, `14.0 s` horizon) and record first-pass task and constraint completion (`all_constraints_passed`).

### 6.1 Train vs. Held-Out Test Split Protocol (`--benchmark`)
Running `python -m evaluation --checkpoint <path> --benchmark --episodes 20` evaluates **`120` closed-loop episodes total**:
1. **Training Split (`60` episodes, `20` per task)**: Replays the exact scene seeds stored in `data/sim_demos.h5` (`seeds 1000..1549` for Task 0, `2000..2549` for Task 1, `3000..3549` for Task 2) to measure in-distribution closed-loop execution accuracy.
2. **Held-Out Test Split (`60` episodes, `20` per task)**: Evaluates unseen test seeds (`seeds 9000..9019` for Task 0, `10000..10019` for Task 1, `11000..11019` for Task 2) where all 4 objects spawn at novel tabletop positions never seen during training.

### 6.2 Automated Failure-Stage Diagnosis & Multi-Camera HUD GIFs
Every episode returns an `EpisodeEvalResult` tracking closest pinch-to-source XY distance (`min_pinch_src_xy_cm`), maximum object lift (`max_src_lift_cm`), final object-to-target XY distance (`final_src_tgt_xy_cm`), bystander/target disturbance checks, and an automated failure classification (`PASS`, `Missed source reach`, `Failed grasp/lift`, `Missed target`, or `Bystander/Target disturbed`). When `--demos` or `--save-gif` is enabled, `RolloutVisualizer` saves both an animated multi-camera `.gif` and a 6-keyframe `.png` strip overlaying projected `[SRC]` and `[TGT]` markers, the predicted 16-step trajectory ribbon, and live joint telemetry.

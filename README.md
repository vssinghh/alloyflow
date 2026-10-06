# AlloyFlow

**AlloyFlow** trains a single multi-task Vision Flow Matching policy for the 6-DoF **SO-ARM101** robot arm (`5` arm joints + `1` gripper) from 3 synchronized `128x128` RGB cameras. It evaluates how a single neural network learns 3 tabletop manipulation tasks across 4 training regimes: `Sim-Only`, `Real-Only`, `Sim -> Real Finetune`, and `Sim + Real Co-Training`.

***

## 1. The 3 Tabletop Tasks
All 3 tasks run on the same tabletop scene with 4 everyday objects (`pen_holder`, `cup`, `bowl`, `rubiks_cube`) and are selected by passing a discrete task ID (`0`, `1`, or `2`):

| Task ID | Task Name | Goal | What It Tests |
| :---: | :--- | :--- | :--- |
| **`0`** | `pick_pen_holder_to_bowl` | Pick up the pen holder and place it into the bowl | Baseline pick and place |
| **`1`** | `pick_cup_to_bowl` | Ignore the pen holder, pick up the cup, and place it into the bowl | Selecting a different source object for the same goal |
| **`2`** | `stack_cup_on_cube` | Pick up the cup and stack it on top of the Rubik's cube | Routing the same source object to a high-precision target |

***

## 2. The 4 Training Modes

| Mode | CLI Flag | Training Data | What We Measure |
| :--- | :--- | :--- | :--- |
| **Mode 1: `Sim-Only`** | `--mode sim_only` | `300` MuJoCo demos (`100`/task) | Sim baseline and zero-shot Sim-to-Real transfer gap |
| **Mode 2: `Real-Only`** | `--mode real_only` | `60` real SO-ARM101 demos (`20`/task) | Real-world learning from small data and failure modes on new positions |
| **Mode 3: `Finetune`** | `--mode finetune` | Pretrain on `300` Sim, finetune on `60` Real | Real-world adaptation vs. forgetting simulation skills |
| **Mode 4: `Co-Train`** | `--mode cotrain` | `50%` Sim + `50%` Real in every batch | Best combined real-world generalization and simulation retention |

***

## 3. Quickstart Commands

```bash
# Install dependencies
uv sync

# 1. Collect demonstrations (Sim or Real)
uv run python -m collection --domain sim --task all --episodes 100
uv run python -m collection --domain real --task 0 --episodes 20

# 2. Train policy (Colab GPU or local Mac MPS; modes: sim_only | real_only | finetune | cotrain)
./scripts/train.sh exp07_dart_full --colab --mode sim_only
./scripts/train.sh exp07_dart_full --local --mode sim_only
uv run python -m training --mode cotrain --real-ratio 0.5

# 3. Evaluate policy (120-episode benchmark or per-demo diagnostic GIFs)
uv run python -m evaluation --checkpoint checkpoints/exp07_dart_full/best_policy.pt --benchmark --episodes 20
uv run python -m evaluation --checkpoint checkpoints/exp07_dart_full/best_policy.pt --demos demo_0000,demo_0101,demo_0200

# 4. Run unit tests
uv run pytest
```

***

## 4. Documentation
* **[`DESIGN.md`](./DESIGN.md)**: System architecture, repository layout, data collection constraints, and HDF5 format.
* **[`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md)**: Master benchmark table and run-by-run findings across all 3 tasks and 4 modes.

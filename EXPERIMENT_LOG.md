# AlloyFlow: Experiment Log

This log tracks every experiment run in **AlloyFlow**. Each entry records what we tested, the exact dataset and training mode used, the success rates across all 3 tasks (`Task 0: Pen Holder to Bowl`, `Task 1: Cup to Bowl`, `Task 2: Stack Cup on Cube`), and what we learned in plain English.

***

## 1. Master Results Table

| Run ID | Training Mode | Dataset Used | Sim Success (`T0 / T1 / T2 / Avg`) | Real Success (`T0 / T1 / T2 / Avg`) | Step Time | Main Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Run 0 (Pilot)** | `Sim-Only` (Clean) | `50` Clean Sim demos/task (`150` total) | `TBD` | `Not run` | `TBD` | Verify 6-DoF planner and 3-task policy in Clean Sim |
| **Run 1** | `Mode 1: Sim-Only` | `100` Sim demos/task (`50 Clean + 50 DR`, `300` total) | `TBD` | `TBD` (Zero-Shot) | `TBD` | Baseline Sim accuracy and zero-shot gap on real robot |
| **Run 2** | `Mode 2: Real-Only` | `20` Real demos/task (`60` total) | `Not run` | `TBD` | `TBD` | How well the real robot learns from small real data alone |
| **Run 3** | `Mode 3: Finetune` | Start from `Run 1`, finetune on `60` Real demos | `TBD` | `TBD` | `TBD` | Does finetuning improve Real accuracy or forget Sim skills? |
| **Run 4** | `Mode 4: Co-Train` | `50%` Sim (`300` demos) + `50%` Real (`60` demos) per batch | `TBD` | `TBD` | `TBD` | Does mixing Sim and Real in every batch work best overall? |

***

## 2. Experiment Entries (Template)

### Run 0: Clean Simulation Pilot (`Sim-Only`)
* **Goal**: Verify that our scripted 6-DoF motion planner and 3-camera policy can solve all 3 tasks in clean simulation before adding domain randomization or real robot data.
* **Setup**:
  - **Mode**: `sim_only`
  - **Data**: `50` clean demos per task (`150` total) in `data/alloy_sim_clean_150.h5`
  - **Checkpoint**: `checkpoints/run0_pilot/best_policy.pt`
* **Results**:
  - **Task 0 (`pick_pen_holder_to_bowl`)**: `TBD`
  - **Task 1 (`pick_cup_to_bowl`)**: `TBD`
  - **Task 2 (`stack_cup_on_cube`)**: `TBD`
  - **Overall Sim Success**: `TBD`
* **What We Learned**:
  - *To be filled after Run 0 completes.*
* **Next Step**:
  - *To be filled after Run 0 completes.*

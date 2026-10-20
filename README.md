# Tunable fingertip compliance for a UR5 + Wuji hand (MSc thesis)

Simulation code for the MSc thesis of Balázs Peisz (EPFL): imitation learning for a **UR5 arm with a 20-DoF Wuji right hand** whose joints are position-controlled only. A **fingertip admittance controller** makes the stiffness of every fingertip tunable at runtime, so that a policy can later choose not only where the fingers go but also how stiff they are.

Built as an Isaac Lab extension (from the Isaac Lab project template).

## Contents

- [Requirements and installation](#requirements-and-installation)
- [Repository layout](#repository-layout)
- [Tasks](#tasks)
- [How the fingertip admittance works](#how-the-fingertip-admittance-works)
- [Running the scripted lift experiment](#running-the-scripted-lift-experiment)
- [Plotting the results](#plotting-the-results)
- [Recording videos](#recording-videos)
- [Tests and tools](#tests-and-tools)
- [Notes and pitfalls](#notes-and-pitfalls)
- [IDE setup, formatting and troubleshooting](#ide-setup-formatting-and-troubleshooting)

## Requirements and installation

Developed and tested with:

- Isaac Sim 5.1, Isaac Lab 2.3 (`isaaclab` 0.54.4)
- Ubuntu 24.04, NVIDIA RTX 5090
- a conda environment with Isaac Lab installed (here `env_isaaclab`)

Install Isaac Lab by following its [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), clone this repository **outside** the Isaac Lab directory, then install the extension in editable mode:

```bash
conda activate env_isaaclab
cd tunable_comp_thesis
python -m pip install -e source/tunable_comp_thesis
```

Check that the tasks register:

```bash
python scripts/tools/check_task_import.py   # prints the registered UR5Wuji task IDs
```

The plot script only needs `numpy` and `matplotlib` and runs without Isaac Sim. Video recording needs `moviepy` (`pip install moviepy` if it is missing).

## Repository layout

```
scripts/
  experiments/
    scripted_lift.py           scripted lift experiment (grasp, K sweep, logging, video)
    plot_scripted_lift.py      all figures from the experiment logs
  tools/
    test_admittance.py         step response of the dynamic controller alone
    test_admittance_variants.py   dynamic vs static controller (open loop, toy contact)
    test_fingertip_admittance.py  the action term inside the env
    test_scripted_lift_env.py  does the arm reach the commanded grasp-point poses?
    probe_hand_kinematics.py   fingertip positions / pad directions for a grid of thumb poses
    check_task_import.py       do the tasks register?
  inspect_ur5.py, check_usd.py, dump_drives.py, dump_prims.py   USD / articulation inspection
  rsl_rl/, skrl/, zero_agent.py, random_agent.py, list_envs.py  from the Isaac Lab template

source/tunable_comp_thesis/
  data/                        USD assets: UR5, Wuji hand (left/right), combined ur5_wuji_{right,left}.usd
  tunable_comp_thesis/
    assets/                    robot configs: UR5_CFG, WUJI_HAND_*_CFG, UR5_WUJI_RIGHT_CFG / _LEFT_CFG
    controllers/
      admittance_controller.py         dynamic law  M e'' + D e' + K e = f
      static_admittance_controller.py  static law   e = f / K
    tasks/manager_based/lift_compliant/
      lift_env_cfg.py          lift scene and MDP (adapted from Isaac Lab's Franka lift task)
      mdp/fingertip_admittance_action.py   the fingertip admittance action term
      config/ur5_wuji/
        joint_pos_env_cfg.py         UR5 + Wuji lift env (robot, hand action, cube, contact sensors)
        scripted_lift_env_cfg.py     scripted variant (arm in task space, deterministic scene)
        __init__.py                  gym registrations
        agents/rsl_rl_ppo_cfg.py     PPO runner config
```

`tasks/direct/...` and `tasks/manager_based/tunable_comp_thesis/...` are unused leftovers of the template.

## Tasks

| Task ID | Environment config | Purpose |
|---|---|---|
| `Isaac-Lift-Cube-UR5Wuji-v0` | `joint_pos_env_cfg:UR5WujiCubeLiftEnvCfg` | Lift task for learning: arm = joint position offsets, hand = fingertip admittance |
| `Isaac-Lift-Cube-UR5Wuji-Play-v0` | `joint_pos_env_cfg:UR5WujiCubeLiftEnvCfg_PLAY` | Same, 16 envs, no observation noise |
| `Isaac-Lift-Cube-UR5Wuji-Scripted-v0` | `scripted_lift_env_cfg:UR5WujiCubeScriptedLiftEnvCfg` | For scripts: arm = absolute grasp-point pose through differential IK, hand = absolute joint references, no randomization, friction μ = 1 |

The scene: the robot on a table, a 42 mm cube (Isaac's DexCube scaled 0.8, 0.216 kg) at (0.5, 0, 0.021) m in the robot base frame. Physics runs at 100 Hz, the environment at 50 Hz (decimation 2).

## How the fingertip admittance works

The policy or script sends 20 finger joint references $q_{ref}$. Every physics step the action term (`FingertipAdmittanceAction`):

1. reads the contact force $f$ from the cube on each of the 5 fingertips;
2. computes an offset $e$ per fingertip with an admittance law;
3. maps it to the finger joints with the damped pseudo-inverse of the fingertip Jacobian, $\Delta q = J^\top (J J^\top + \lambda^2 I)^{-1} e$, $\lambda = 0.01$;
4. sends $q_{cmd} = q_{ref} + \Delta q$ (within the joint limits) to the joint drives.

Two admittance laws are available (`controller=AdmittanceControllerCfg()` or `StaticAdmittanceControllerCfg()`):

| | Law | Notes |
|---|---|---|
| Dynamic | $M\ddot e + D\dot e + Ke = f_{filt}$, $D = 2\sqrt{KM}$, $M = 0.3$ kg | critically damped; stable for $K \lesssim 2059$ N/m at $dt = 0.01$ s |
| Static | $e = f_{filt}/K$ | stable through a stiff contact only for $K > K_d / (2/\alpha - 1)$ |

Both low-pass the force ($\alpha = 0.3$), limit $\lvert e\rvert$ to 30 mm, and treat $K = \infty$ as "admittance off". At steady state both give $e = f/K$. Together with the joint drives (stiffness $K_d$ at the fingertip) the fingertip renders $K_r = (1/K + 1/K_d)^{-1}$: the admittance can only make a fingertip softer.

The stiffness is set per environment with `stiffness_values` (env $i$ gets `stiffness_values[i % len]`).

## Running the scripted lift experiment

`scripts/experiments/scripted_lift.py` grasps the cube with the thumb on one face and fingers 2–5 on the opposite face, lifts it 15 cm and holds it. Every environment gets its own stiffness, so one run compares several K:

```bash
# one env per stiffness, dynamic or static admittance (inf = admittance off)
python scripts/experiments/scripted_lift.py --headless --controller dynamic --stiffness 1 50 75 100 150 500 inf
python scripts/experiments/scripted_lift.py --headless --controller static  --stiffness 1 50 75 100 150 500 inf
```

Leave out `--headless` to watch it in the GUI.

**Motion (11 s, logged at 50 Hz):** settle 0.5 s → approach 2 s (10 cm above the grasp) → descend 1.5 s → close 2 s → squeeze 1 s → lift 2 s (15 cm) → hold 2 s. Before that (not logged) the cube settles, the script measures where the fingertips are and places the grasp so that the cube sits between thumb and fingers.

**Grasp parameters.** The defaults are the tuned grasp; every per-env argument takes one value for all envs or one value per env (e.g. `--thumb_curl_closed 0.9 0.7`):

| Argument | Default | Meaning |
|---|---|---|
| `--stiffness` | `50 150 500 inf` | K per env [N/m] |
| `--controller` | `dynamic` | `dynamic` or `static` |
| `--fingers_closure` | 0.7 | flexion of fingers 2–5 (joints 1, 3, 4) when closed [rad] |
| `--index_middle_extra` | 0.05 | index and middle close this much further [rad] |
| `--adduction` | 0.12 | spread: ring +a, little +2a toward the middle finger [rad] |
| `--index_adduction` | 0.0 | spread of the index toward the middle finger [rad] |
| `--thumb_j1`, `--thumb_j2`, `--thumb_curl` | 1.0, −0.14, 0.9 | thumb pre-shape (joints 3 and 4 = curl) |
| `--thumb_curl_closed` | 0.7 | thumb pushes into the cube while closing |
| `--thumb_j2_closed` | = `--thumb_j2` | thumb joint2 at the end of the closing |
| `--grasp_dx/dy/dz` | measured | override the grasp point relative to the cube centre [m] |

**Output.** A summary is printed at the end: lift height, cube rotation (roll/pitch/yaw), fingertip forces and contact positions. Its parameter columns show what actually ran. Everything is saved to `logs/scripted_lift/<controller>_<date>_<time>.npz`. Arrays are time-first, `(T, N, ...)`:

| Field | Shape | Content |
|---|---|---|
| `t`, `phase` | (T,) | time [s], phase index (names in `phase_names`) |
| `tip_force` | (T, N, 5, 3) | contact force from the cube on each fingertip [N] |
| `tip_offset` | (T, N, 5, 3) | admittance offset $e$ [m] |
| `tip_pos`, `tip_pad` | (T, N, 5, 3) | fingertip position [m], pad normal (base frame) |
| `q`, `q_ref`, `q_cmd` | (T, N, 20) | hand joint position, reference, command [rad] |
| `tau_pd`, `tau_applied` | (T, N, 20) | hand joint torque from the PD law, and Isaac Lab's value [Nm] |
| `tip_ref_offset`, `tip_cmd_offset` | (T, N, 5, 3) | $J(q_{ref}-q)$, $J(q_{cmd}-q)$: how far reference/command reach into the cube [m] |
| `q_arm` | (T, N, 6) | arm joints [rad] |
| `cube_pos`, `cube_quat` | (T, N, 3/4) | cube position (base frame) [m], orientation (w, x, y, z) |
| `ee_pos`, `ee_target` | (T, N, 3) | grasp point and its target (base frame) [m] |

Plus the per-env parameters, `stiffness`, `controller`, `joint_names`, `dt`, `cube_start`, `cube_mass`, `grasp_offset`, `valid` (arm reached its start pose) and the hand drive gains `joint_stiffness`, `joint_damping`.

## Plotting the results

```bash
python scripts/experiments/plot_scripted_lift.py \
    logs/scripted_lift/dynamic_<time>.npz logs/scripted_lift/static_<time>.npz --pdf
```

Figures go to `logs/scripted_lift/plots/` (`--out` to change; `--pdf` also saves PDFs). Per run:

| Figure | Shows |
|---|---|
| `<run>_overview` | cube height, thumb force, summed finger force, thumb offset, cube rotation |
| `<run>_fingers` | contact force and offset of every fingertip |
| `<run>_joints` | $q_{ref}$, $q_{cmd}$ and $q$ of thumb joint3 and one finger's joint1 (`--finger`) |
| `<run>_torque` | joint3 torque of every finger over time and against K |
| `<run>_force_vs_angle` | contact force against $\lvert q_{ref}-q\rvert$ |
| `<run>_force_vs_position` | contact force against $\lVert J(q_{ref}-q)\rVert$, with the predicted $K_r$ |

With runs of both controllers also `compare_controllers` and `compare_torque`. A summary table is printed as well.

## Recording videos

Add `--video --video_env i` to the same command; it records environment `i` for the whole motion:

```bash
python scripts/experiments/scripted_lift.py --headless --controller static \
    --stiffness 1 50 75 100 150 500 inf --video --video_env 3
```

- Output: `logs/scripted_lift/videos/<controller>_<time>_env<i>_K<k>/` (videos are git-ignored).
- The env number follows the order of `--stiffness` (above: env 0 = K 1, …, env 6 = rigid).
- Camera: `--cam_eye x y z` (default `0.85 0.45 0.30`) and `--cam_lookat x y z` (default `0.5 0.0 0.08`), relative to the env origin.
- One run records one env. With the same `--stiffness` list the physics is identical to the run without video, so the video shows exactly the logged run.

All envs of both controllers:

```bash
for c in dynamic static; do
  for i in 0 1 2 3 4 5 6; do
    python scripts/experiments/scripted_lift.py --headless --controller $c \
      --stiffness 1 50 75 100 150 500 inf --video --video_env $i
  done
done
```

## Tests and tools

```bash
python scripts/tools/test_admittance.py --headless      # dynamic controller: steady state, overshoot, stability, clamp
python scripts/tools/test_admittance_variants.py --headless   # dynamic vs static, open loop and through a toy contact
python scripts/tools/test_fingertip_admittance.py --headless --controller dynamic   # action term in the env
python scripts/tools/test_scripted_lift_env.py --headless   # arm tracking of the grasp point
python scripts/tools/probe_hand_kinematics.py --headless    # fingertip map for a grid of thumb poses
python scripts/inspect_ur5.py --headless                    # joints, bodies, drives of UR5 + Wuji
```

## Notes and pitfalls

- **Contact sensors.** `fingertip_contacts` (one sensor on all five tip links) also measures contacts with other fingers, the palm or the table. The admittance therefore reads five extra sensors `tip{1..5}_object_contact`, each filtered to the cube. PhysX needs one sensor per fingertip for this: one filtered sensor on all five tips does not work and reads zero. Contact sensors require `activate_contact_sensors=True` on the robot.
- **Normal forces only.** The sensors report the normal contact force, not friction; the admittance reacts to the squeeze, not to the weight of the cube.
- **The 30 mm offset limit.** For soft K the offset hits the limit when the cube is lifted; from then on the fingertip is effectively rigid at a retracted position (`max_offset` in the controller configs).
- **Determinism.** With the same number of environments and settings a run is reproducible. A different number of envs changes the results slightly (contact solver order), so compare stiffness values within one run.
- **Prim paths vs names.** Sensors take USD prim paths (`{ENV_REGEX_NS}/Robot/WujiHand/right_finger1_tip_link`); joint and body names take articulation names (no `WujiHand` in between).
- **Hand joint order** is breadth-first: `right_finger1_joint1, right_finger2_joint1, …, right_finger5_joint1, right_finger1_joint2, …`. Finger 1 is the thumb; joint2 of fingers 2–5 is the spread joint. The hand drives use the system-identified gains from the Wuji USD ($k_p = 2$ Nm/rad, $k_d = 0.05$ Nms/rad).
- **Git LFS.** `.gitattributes` (from the template) routes `*.usd` and other large formats through Git LFS, but the USD files are stored as ordinary files. If they show as modified right after cloning, that is why; remove the `*.usd` / `*.usda` lines from `.gitattributes` to fix it.
- **`scripts/list_envs.py`** still filters task IDs for the template prefix `Template-`, so it does not list the UR5Wuji tasks; use `scripts/tools/check_task_import.py` instead.

## IDE setup, formatting and troubleshooting

**VS Code:** run the task `setup_python_env` (`Ctrl+Shift+P` → `Tasks: Run Task`) and give the path to your Isaac Sim installation. It creates `.vscode/.python.env` with the paths of the Isaac Sim / Omniverse extensions for code completion.

**Formatting:** `pip install pre-commit`, then `pre-commit run --all-files`.

**Pylance does not index the extension:** add it to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": ["<path-to-repo>/source/tunable_comp_thesis"]
}
```

**Pylance crashes (too many files indexed):** comment out unused Omniverse packages under `"python.analysis.extraPaths"` in `.vscode/settings.json`, e.g. `omni.anim.*`, `omni.kit.*`, `omni.graph.*`, `omni.services.*`.
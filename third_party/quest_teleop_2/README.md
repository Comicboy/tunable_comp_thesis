# quest_teleop_2

Teleoperate a UR5e and a Wuji Hand 2 from a Meta Quest 3, and record demonstrations in HDF5
for ACT. Taken from `quest_teleop` (teleoperation, the scripted reach, recording, calibration,
safety), plus a tactile glove on the robot hand whose contact forces can drive the hand
through an admittance controller while you teleoperate.

## Environments

| Environment | Used for | Notes |
|---|---|---|
| `teleop` | everything in `scripts/` | Python 3.11, ur_rtde, wuji-sdk, wuji_retargeting |
| `wuji-glove` | reading the glove (`scripts/glove/glove_server.py`) | Python 3.12, wujihandpy 1.8.0; the teleop scripts start it themselves |
| `sam3` | finding the object for the scripted reach | started on demand, as in quest_teleop |

Run every command from this folder with the `teleop` environment active.

The glove needs wujihandpy 1.7 or later, which cannot share the `teleop` environment, so it
is read in its own process and streamed to the teleop process over a local socket. To build
that environment elsewhere:

```bash
conda create -y -n wuji-glove python=3.12 numpy pip
~/miniforge3/envs/wuji-glove/bin/python -m pip install wujihandpy==1.8.0 opencv-python==4.11.0.86
```

## Hardware

- **UR5e** at the IP in `config/robot.yaml`, in Remote Control. Workspace box, speed caps and
  the 30 N force guard are in the same file and apply to every script.
- **Wuji Hand 2** over Ethernet. It has no emergency stop: the robot's e-stop does not cut it.
- **Quest 3** over USB-C.
- **Cameras**: the top-down one finds the object; the wrist one is what the policy sees
  (`config/cameras.yaml`).
- **Tactile glove** on the robot hand, its triangle-marked USB-C port to this PC. Leave the
  circle port empty: it only bridges to a first-generation hand.

## Teleoperate

```bash
bash setup/quest_usb.sh                      # after every replug or headset sleep
python scripts/teleop_all.py --hand-current 1.0
```

Open `http://localhost:8080` in the Quest browser and enter VR. SPACE engages; your right
hand drives the arm and the fingers; a left-hand pinch freezes the arm so you can
reposition. Add `--glove` to read the glove, or `--stiffness K` to run the admittance.

## The glove, once per mounting

1. **Which cells are which finger.** Press along each finger's pads in turn when asked:
   ```bash
   python scripts/glove/glove_regions.py
   ```
   Until you do, `config/glove_regions.yaml` is a guess from the sensor layout.
2. **The noise floor.** The glove reads pressure when the fingers move even without contact.
   Run teleoperation with the glove in one terminal, then in another:
   ```bash
   python scripts/teleop_all.py --hand-current 1.0 --glove
   python scripts/glove/glove_noise.py --seconds 60
   ```
   Keep the hand still and untouched for the first 2 s, then move it through everything a
   demonstration does, touching nothing. Each cell's floor becomes 1.25 x the most it rose; below
   its floor a cell reads exactly zero, and above it only the excess counts. Saved to
   `config/glove_noise.npz` and used whenever that file exists.
3. **Check it.** This shows exactly what the controller sees, alongside teleoperation:
   ```bash
   python scripts/glove/glove_view.py
   ```

Every run learns the glove's rest during its first 2 s, so nothing may touch the glove then.
`g` learns it again between episodes.

## Record

```bash
python scripts/record_grasps.py --out data/admittance/k50.hdf5 --object cube \
    --hand-current 1.0 --stiffness 50
```

`t` homes, finds the cube, reaches it with a deliberate random error and starts recording.
SPACE engages, you grasp, and `s` saves (`f` saves a failure, `r` discards, `g` relearns the
glove's rest, `q` quits). An episode will not start until the glove's rest is learned.
Record one file per stiffness (15 demos each). Put the cube on the same spot or two each time.
`--glove` without `--stiffness` records the glove with the hand following you unchanged.

Check a file, and replay a demo on the robot (open loop, with the cube where it was):

```bash
python scripts/data/inspect_dataset.py --dataset data/admittance/k50.hdf5
python scripts/replay_demo.py --dataset data/admittance/k50.hdf5 --demo 0 --hand-current 1.0
```

## What a file holds

Per episode `data/demo_N`, at 25 Hz:

| Key | Shape | |
|---|---|---|
| `actions` | (T, 26) | 6 arm joint targets (rad) + 20 hand joints sent to the hand (normalised 0..1): the admittance's command |
| `obs/joint_pos`, `obs/joint_vel` | (T, 6) | measured arm |
| `obs/gripper` | (T, 20) | measured hand, normalised |
| `obs/images/wrist`, `obs/images/static` | (T,) | JPEG per frame |
| `obs/tactile` | (T, 24, 32) | glove pressure 0..1, NaN outside the sensors |
| `obs/finger_force` | (T, 5) | contact force per finger, thumb..little |
| `obs/stiffness` | (T, 1) | the stiffness of this episode (inf without admittance) |
| `extra/hand_reference` | (T, 20) | your retargeted hand, before the admittance |
| `extra/hand_admittance` | (T, 20) | the admittance's command (same as the hand part of `actions`) |
| `extra/palm_force` | (T, 1) | palm contact force, recorded only |
| `extra/tcp_pose`, `extra/target_tcp_pose`, ... | | diagnostics, as in quest_teleop |

Episode attributes include `stiffness`, `admittance_mass`, `admittance_damping`,
`admittance_weights`, `admittance_law`, `glove_serial`, `glove_noise_floor`, and the reach's
`object_xy_m`, `approach_noise_m`, `arrived_tcp_pose`.

## The admittance

`quest_teleop2/control/admittance.py`. Per finger f, on its four joints, with e the command
minus your reference (normalised joint units, 1 = a joint's full range) and F_f the finger's
contact force:

    M e'' + D e' + K e = -F_f w_f,      q_cmd = clip(q_ref + e, 0, 1)

`w_f` spreads the force over the finger's joints (flexion joints by default, abduction not),
so a contact opens the finger against your reference by F/K at rest. M = 0.2 and critical
damping by default, |e| at most 0.4 per joint (`config/admittance.yaml`; `--mass`,
`--damping` and `--stiffness` override it). The force is never negative, so the admittance
only ever opens the hand relative to your reference.

F is the glove's pressure above its rest and above the noise floor, summed over the finger's
cells in contact (`quest_teleop2/tactile/contact.py`): unitless, so K is in those units per
joint range. A firm fingertip contact sums to a few units; start around K = 20-100.

**Another law** plugs in without touching the rest: a class built from `AdmittanceParams`,
with `reset()` and `step(q_ref, force, dt) -> q_cmd` (20 normalised joints thumb..little,
5 forces), passed as `--admittance-law module:Class`.

## Tests

```bash
python -m pytest tests -q
```

No hardware needed: the session, recording, safety layer, admittance, glove link and contact
detection run against fakes.

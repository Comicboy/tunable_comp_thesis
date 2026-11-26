# Scripts

Run everything from this folder with the `teleop` environment active (`python ...`). The two
exceptions start themselves when needed: the SAM 3 server (`sam3` environment) and the glove
server (`wuji-glove` environment). Every script has `--help`.

## Flags shared by many scripts

| Flag | Meaning |
|---|---|
| `--yes` | skip the "Proceed?" confirmation |
| `--hand-current A` | hand current limit per joint, amps. Driver default 0.5 (weak); use 1.0 |
| `--serial S` | hand serial, only if more than one hand is connected |
| `--left` | the robot is wearing a LEFT hand |
| `--no-human-limits` | let the fingers hyperextend to the URDF limits (off: they stop where a human finger does) |
| `--native-left-solver` | with `--left`, use the vendor's left retargeter instead of mirroring the right one |
| `--urdf PATH` | hand URDF for the joint limits (default: the Hand 2's own) |
| `--object NAME` | what SAM 3 looks for: a name in `config/objects.yaml` (e.g. `cube`) or a short phrase in quotes |
| `--sam / --no-sam` | find the object with SAM 3 (default) or the old background detector |
| `--object-height M` | object height, needed when `--object` is a phrase not in `objects.yaml` |
| `--approach-from-above` | reach by crossing at home height and descending straight down |
| `--sam-pick one\|best` | when several objects match equally: refuse (default) or take the best |
| `--robot-config`, `--teleop-config`, `--task-config` | use another config file than `config/robot.yaml` etc. |
| `--host IP` | robot IP, instead of the one in `robot.yaml` |
| `--port N` | port of the headset page (default 8080) |
| `-v` | debug logging |

Glove and admittance flags, on `teleop_all.py` and `record_grasps.py`:

| Flag | Meaning |
|---|---|
| `--glove` | read the glove and record its pressure and per-finger forces; the hand follows you unchanged |
| `--stiffness K` | run the admittance with stiffness K (implies `--glove`) |
| `--mass M`, `--damping D` | override `config/admittance.yaml` (default M 0.2, critical damping) |
| `--admittance-config PATH` | another admittance config file |
| `--admittance-law MODULE:CLASS` | another law, e.g. `student_law:StudentAdmittance` |
| `--glove-regions PATH` | which glove cells are which finger (default `config/glove_regions.yaml`) |
| `--glove-noise PATH` | per-cell noise floor (default `config/glove_noise.npz`, used if present) |
| `--glove-python PATH` | python of the environment that reads the glove |

## Setup and checks

**`bash setup/quest_usb.sh [PORT]`**: forwards the headset's `localhost:8080` to this PC over
USB-C. Run it after every replug or headset sleep.

**`python setup/preflight.py [--skip-robot]`**: checks this machine, the headset link and (unless
`--skip-robot`) the robot connection. Moves nothing.

**`python scripts/troubleshooting/check_linux_stack.py [--skip-hand]`**: checks the Linux-specific
parts (RTDE, cameras, hand). `--skip-hand` when the hand is not attached. Moves nothing.

**`python scripts/troubleshooting/quest_button_test.py`**: shows what the headset sends. No robot.
`--write` saves the measured values to `teleop.yaml`; `--seconds N` exits after N s.

**`python -m quest_teleop2.vision.camera --list`**: lists the cameras and their serials.

**`python -m pytest tests -q`**: the offline tests; no hardware.

## Teleoperation

**`python scripts/teleop_all.py`**: arm and hand from the headset, nothing recorded. Moves.

| Flag | Meaning |
|---|---|
| `--no-hand` | arm only, hand de-energised |
| `--no-home` | do not move to home at startup |
| `--cutoff HZ` | hand low-pass cutoff (default 5) |
| `--optimizer` | ignored on the Hand 2 (always the adaptive solver) |
| + hand, glove and config flags above | |

Keys: SPACE engage/disengage, `a` re-anchor, `h` home, `g` relearn the glove's rest, `q` quit.
A left-hand pinch freezes the arm.

**`python scripts/troubleshooting/teleop_ur5e.py`**: arm only. Moves. `--no-home`, `--yes`.

**`python scripts/troubleshooting/teleop_hand.py`**: hand only, no arm; a dry run unless `--run`.

| Flag | Meaning |
|---|---|
| `--run` | actually drive the hand |
| `--rate HZ`, `--cutoff HZ` | control rate and low-pass |
| `--calibrate` | measure where your relaxed hand lands, for `config/hand_calibration.yaml`; never energises the hand |
| `--samples N` | frames averaged by `--calibrate` (default 150) |
| `--record FILE.npz` | save the retargeted 20-D commands |

## Recording

**`python scripts/record_grasps.py --out FILE.hdf5`**: the recorder for this project: a scripted reach
to the object, then you teleoperate the grasp. Moves.

| Flag | Meaning |
|---|---|
| `--out FILE` | HDF5 file; an existing one is appended to (default `data/grasp/cube.hdf5`) |
| `--object NAME` | what SAM 3 finds, also stored on every episode (default `cube`) |
| `--standoff M` | how far above the taught height the reach stops (default 0.04, or as the file) |
| `--noise-xy M` | random arrival error, +-M in x and y (default 0.04, or as the file) |
| `--noise-z M` | random arrival error, 0..M upward only (default 0.02, or as the file) |
| `--seed N` | seed of that random error |
| `--cameras a,b` | which cameras to record (default all, in `cameras.yaml` order) |
| `--record-hand / --no-record-hand` | also store your raw tracked hand joints (default on) |
| `--from-home` | no reach: start at home and teleoperate everything |
| `--goal-xy X,Y ...` | goal positions, cycled with `]`, stored per episode |
| `--cutoff HZ` | hand low-pass |
| + hand, SAM, glove and config flags above | |

Keys: `t` reach and start recording, SPACE engage, `s` save, `f` save as a failure, `r` discard,
`h` home, `g` relearn the glove's rest, `q` quit.

**`python scripts/record_demos.py --out FILE.hdf5`**: the older recorder: whole task from home, with
start/goal labels. No glove support. `--no-hand`, `--start N`, `--goal N` (initial labels),
`--teach` (capture start/goal poses into `task.yaml`), `--cameras` (`none` for state only),
`--no-auto-home`, `--no-home`, `--record-hand`, `--cutoff`.

## The glove

**`python scripts/glove/glove_view.py`**: live view of what the admittance sees: contacts above
the noise floor, finger regions, force bars. Runs alongside teleoperation. Keys: `b` relearn the
rest, `v` raw/contacts, `+ -` colours, `f m r` flip, mirror, rotate, `q` quit.
Flags: `--glove-regions`, `--glove-noise`, `--glove-python`.

**`python scripts/glove/glove_regions.py`**: maps glove cells to fingers: press each finger's pads
in turn. `--seconds S` per finger (default 6), `--out` (default `config/glove_regions.yaml`).

**`python scripts/glove/glove_noise.py`**: learns the noise floor from the hand moving without
contact (run `teleop_all.py --glove` alongside). `--seconds S` of motion (default 60),
`--margin X` (floor = X times the largest rise, default 1.25), `--minimum F` (smallest floor,
default 0.01), `--out` (default `config/glove_noise.npz`).

**`~/miniforge3/envs/wuji-glove/bin/python -I scripts/glove/glove_server.py`**: the glove reader.
Started automatically; run by hand only to debug. `--socket PATH`, `--serial S`.

## The robot

**`python scripts/go_home.py`**: moves the arm home. Moves.
`--speed RAD_S` (default `home_speed`), `--hand` (also park the fingers, then de-energise),
`--dry-run` (moves nothing; prints the hand's parking ramp), + hand flags.

**`python scripts/troubleshooting/set_home.py`**: saves the arm's current joints as home.
`--show` prints without writing.

**`python scripts/troubleshooting/stream_state.py`**: live robot state, read only.
`--seconds N`, `--check-rate` (loop timing).

**`python scripts/troubleshooting/hand_test.py`**: talks to the hand. Read only unless `--move` or `--fingers`.

| Flag | Meaning |
|---|---|
| `--watch` | live joint readout |
| `--normalise` | show 0..1 instead of radians |
| `--move` | drive one joint (`--finger 0-4`, `--joint 0-3`) |
| `--fingers` | one joint on each finger in turn |
| `--amplitude RAD`, `--period S`, `--cycles N`, `--rate HZ` | the motion |

**`python scripts/troubleshooting/jog_test.py`**: small sine on one TCP axis through the safety
layer. Moves. `--axis x|y|z`, `--amplitude M` (default 0.02), `--period S`, `--cycles N`,
`--no-home`, `--test-watchdog` (also checks the controller stops when commands stop).

**`python scripts/troubleshooting/force_guard_test.py`**: moves the arm down and up to test the
30 N stop (against a scale). Moves. `--depth M` (0 holds still), `--period S`, `--cycles N`.

**`python scripts/troubleshooting/servo_hold_test.py`**: for protective stops: holds still, then
optionally creeps one joint. Moves. `--seconds S`, `--nudge RAD_S`, `--joint N`, `--no-hand`,
`--no-home`.

## Cameras and the reach

**`python scripts/camera/calibrate_table.py`**: maps top-camera pixels to table positions, using
SAM 3. Never moves the arm (you freedrive it). Keys: `c` capture the cube's pixel, `t` capture the
TCP over it, `p` store the approach attitude and height, `u` undo, `s` solve and save, `q` quit.
`--camera` (default static), `--out` (default `config/table.yaml`), `--scale`, `--object`,
`--sam / --no-sam`, `--template-only` (recapture only the approach pose).

**`python scripts/troubleshooting/scripted_grasp.py --reach-only`**: tests the reach. Moves.

| Flag | Meaning |
|---|---|
| `--reach-only` | approach and report, never touching the hand |
| `--wait` | wait at home for ENTER before each reach (move the cube), hold over the cube until ENTER |
| `--trials N` | how many reaches (default 10) |
| `--standoff M` | height above the taught pose (default 0.10; recording uses 0.04) |
| `--offset-x M`, `--offset-y M` | shift the aim (default: the calibration's; `--offset-x 0` centres the TCP on the cube) |
| `--save-offset` | write that offset into `config/table.yaml` |
| `--noise-xy M`, `--noise-z M`, `--seed N` | random arrival error, as in recording |
| `--view` | live camera window with the wrist view |
| `--video [DIR]` | film both cameras |
| `--close-from FILE.hdf5` | without `--reach-only`: also close the hand with a pose taken from demos, and lift |
| + SAM flags above | |

**`python scripts/camera/camera_live.py`**: live camera view for aiming and focusing.
`--serial S`, `--grid N`, `--focus F`, `--exposure E`, `--wb K`, `--tile`, `--info / --no-info`,
`--scale X`, `--out DIR` (snapshots).

**`python scripts/camera/capture_stills.py`**: averaged still images. `--label`, `--camera`,
`--width`, `--height`, `--average N`, `--settle S`, `--out`.

**`~/miniforge3/envs/sam3/bin/python scripts/camera/sam_live.py`**: live SAM 3 view of the top
camera. `--object NAME` (shows the reach's decision), `--prompt TEXT` (repeatable),
`--threshold`, `--camera`, `--once` (save one frame), `--fp32`, `--sam-pick`.
Quit it before starting a robot script: it holds the camera.

**`scripts/camera/sam_server.py`**: the SAM 3 server; started automatically. `--socket PATH`.

**`python scripts/camera/check_scene.py`**: the old background detector's diagnostics, only for
`--no-sam`. `--live`, `--compare`, `--diagnose`, `--separability`, `--null`, and more.

## Data

**`python scripts/data/inspect_dataset.py --dataset FILE.hdf5`**: summary and sanity checks.
`--plot PATH` (per-joint plots), `--contact-sheet PNG` (first/middle/last frame per demo),
`--contact-sheet-demos N`.

**`python scripts/replay_demo.py --dataset FILE.hdf5 --demo N`**: replays a demo on the robot,
open loop. Moves (unless `--dry-run`). `--speed X` (<=1, slower), `--dry-run` (analyse only),
`--no-hand`, `--images` (show recorded cameras), `--with-live` (and the live ones),
`--image-scale X`, `--plot PNG` (recorded vs achieved), + hand flags.

**`python scripts/data/drop_demos.py FILE.hdf5 --demos 3 7`**: removes demos and renumbers.
`--dry-run`, `--no-backup` (default keeps a timestamped backup).

**`python scripts/data/reorder_demos.py FILE.hdf5 --swap A B`** or **`--order ...`**: reorders demos.
`--dry-run`, `--no-backup`.

**`python scripts/data/split_dataset.py FILE.hdf5 --out PART.hdf5 --goal g1`**: moves the demos
of one goal (or `--start s2`) to another file. `--keep` copies instead, `--dry-run`.

**`python scripts/data/merge_datasets.py --out ALL.hdf5 a.hdf5 b.hdf5`**: merges files with the
same schema. `--force` (overwrite `--out`), `--only-success`, `--offline-rl` (adds rewards and
dones).

**`python scripts/data/drop_first_frame.py FILE.hdf5`**: drops the first frame of every demo.
`--dry-run`.

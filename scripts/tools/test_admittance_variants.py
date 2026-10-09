"""Compare the dynamic and the static admittance controller (no scene, only the controllers).

1. Open loop: a 2 N step force, as in test_admittance.py, for the static law.
2. Closed loop through a toy contact: the reference is 10 mm inside a rigid surface
   and the finger drives are a spring K_DRIVE, so the contact force is
   f = K_DRIVE * max(0, overlap - e). Expected steady state for both:
   f = K_r * overlap with K_r = (1/K + 1/K_DRIVE)^-1, if the loop is stable.

    python scripts/tools/test_admittance_variants.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare the admittance controller variants.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from tunable_comp_thesis.controllers import (
    AdmittanceControllerCfg,
    StaticAdmittanceController,
    StaticAdmittanceControllerCfg,
)

DT = 0.01  # [s]
STEPS = 300  # 3 s
FORCE = 2.0  # [N] open-loop step
OVERLAP = 0.010  # [m] reference depth inside the surface
K_DRIVE = 400.0  # [N/m] assumed drive stiffness at the fingertip (a guess; the sim will tell)
STIFFNESS = [50.0, 100.0, 150.0, 500.0]  # [N/m], one env each


def open_loop_static():
    n = len(STIFFNESS)
    ctrl = StaticAdmittanceController(StaticAdmittanceControllerCfg(max_offset=1.0), n, 1, DT, "cpu")
    K = torch.tensor(STIFFNESS).unsqueeze(1)
    f = torch.zeros(n, 1, 3)
    f[..., 2] = FORCE
    hist = torch.stack([ctrl.compute(f, K)[:, 0, 2].clone() for _ in range(STEPS)])
    print("\n1. static law, open loop, 2 N step")
    print(f"{'K [N/m]':>8} {'F/K [mm]':>9} {'e_end [mm]':>11} {'overshoot':>10} {'t90 [s]':>8}")
    for i, k in enumerate(STIFFNESS):
        e, target = hist[:, i], FORCE / k
        overshoot = (e.max().item() - target) / target * 100.0
        t90 = (torch.nonzero(e >= 0.9 * target)[0].item() + 1) * DT
        print(f"{k:8.0f} {target * 1e3:9.2f} {e[-1].item() * 1e3:11.2f} {overshoot:9.1f}% {t90:8.2f}")


def closed_loop(cfg):
    n = len(STIFFNESS)
    ctrl = cfg.class_type(cfg, n, 1, DT, "cpu")
    K = torch.tensor(STIFFNESS).unsqueeze(1)
    f_hist = torch.zeros(STEPS, n)
    f = torch.zeros(n, 1, 3)
    for k in range(STEPS):
        e_z = ctrl.offset[:, 0, 2]
        f[:, 0, 2] = K_DRIVE * torch.clamp(OVERLAP - e_z, min=0.0)  # contact only pushes
        f_hist[k] = f[:, 0, 2]
        ctrl.compute(f, K)
    return f_hist


def main():
    open_loop_static()

    print(f"\n2. closed loop: overlap {OVERLAP * 1e3:.0f} mm, K_drive = {K_DRIVE:.0f} N/m (assumed)")
    static_cfg = StaticAdmittanceControllerCfg()
    k_min = StaticAdmittanceController(static_cfg, 1, 1, DT, "cpu").min_stable_stiffness(K_DRIVE)
    print(f"   static law predicted stable for K > {k_min:.0f} N/m (alpha = {static_cfg.force_filter_alpha})")
    print(f"{'K [N/m]':>8} {'variant':>8} {'f_pred [N]':>11} {'f_end [N]':>10} {'ripple [N]':>11}  status")
    for name, cfg in (("dynamic", AdmittanceControllerCfg()), ("static", static_cfg)):
        f_hist = closed_loop(cfg)
        for i, k in enumerate(STIFFNESS):
            f_pred = OVERLAP / (1.0 / k + 1.0 / K_DRIVE)
            last = f_hist[-50:, i]
            ripple = (last.max() - last.min()).item()
            status = "settled" if ripple < 1e-3 and abs(last[-1].item() - f_pred) < 1e-3 else "OSCILLATES"
            print(f"{k:8.0f} {name:>8} {f_pred:11.3f} {last[-1].item():10.3f} {ripple:11.3g}  {status}")


if __name__ == "__main__":
    main()
    simulation_app.close()
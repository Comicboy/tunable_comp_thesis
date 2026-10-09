"""Step-response test of the admittance controller (no scene, only the controller).

Each environment gets its own stiffness and the same 2 N step force along z.
Checks: steady state e = F/K, no overshoot, K = inf -> e = 0, the stability
bound of the discrete update, the |e| clamp and reset().
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-response test of the admittance controller.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

from tunable_comp_thesis.controllers import AdmittanceController, AdmittanceControllerCfg

DT = 0.01  # [s] physics step of the env
FORCE = 2.0  # [N] step force along z
STEPS = 300  # 3 s
STIFFNESS = [50.0, 150.0, 500.0, math.inf, 2000.0, 2200.0]  # [N/m], one env each


def main():
    n = len(STIFFNESS)
    K = torch.tensor(STIFFNESS).unsqueeze(1)  # (N, 1): one stiffness per env
    f = torch.zeros(n, 1, 3)  # (N, P=1, 3)
    f[..., 2] = FORCE

    # --- 1. step response, clamp disabled to see the pure law -------------------
    cfg = AdmittanceControllerCfg(max_offset=math.inf)
    ctrl = AdmittanceController(cfg, num_envs=n, num_points=1, dt=DT, device="cpu")
    hist = torch.zeros(STEPS, n)
    for k in range(STEPS):
        hist[k] = ctrl.compute(f, K)[:, 0, 2]

    print(f"\nM = {cfg.mass} kg, dt = {DT} s, filter alpha = {cfg.force_filter_alpha}, step force = {FORCE} N")
    print(f"stability bound of the update: K_max = {ctrl.max_stable_stiffness:.0f} N/m\n")
    print(f"{'K [N/m]':>8} {'F/K [mm]':>9} {'e_end [mm]':>11} {'overshoot':>10} {'t90 [s]':>8}  status")
    for i, k_val in enumerate(STIFFNESS):
        e = hist[:, i]
        if not math.isfinite(k_val):
            status = "ok (rigid)" if e.abs().max().item() == 0.0 else "FAIL: should stay 0"
            print(f"{'inf':>8} {0.0:9.2f} {e[-1].item() * 1e3:11.2f} {'-':>10} {'-':>8}  {status}")
            continue
        target = FORCE / k_val
        err_end = abs(e[-1].item() - target)
        if err_end < 1e-3 * target:
            overshoot = (e.max().item() - target) / target * 100.0
            t90 = (torch.nonzero(e >= 0.9 * target)[0].item() + 1) * DT
            print(f"{k_val:8.0f} {target * 1e3:9.2f} {e[-1].item() * 1e3:11.2f} {overshoot:9.1f}% {t90:8.2f}  converged")
        else:
            print(f"{k_val:8.0f} {target * 1e3:9.2f} {e[-1].item() * 1e3:11.3g} {'-':>10} {'-':>8}  DIVERGED")

    # --- 2. reset of a subset of envs -------------------------------------------
    ctrl.reset([0])
    print(f"\nreset env 0: |e_0| = {ctrl.offset[0].norm().item():.1e} m (expected 0), "
          f"|e_1| = {ctrl.offset[1].norm().item() * 1e3:.2f} mm (expected 13.33, untouched)")

    # --- 3. clamp with the default max_offset ---------------------------------------
    cfg = AdmittanceControllerCfg()
    ctrl = AdmittanceController(cfg, num_envs=1, num_points=1, dt=DT, device="cpu")
    for _ in range(STEPS):
        e = ctrl.compute(f[:1], torch.tensor([[50.0]]))
    print(f"clamp: F/K = {FORCE / 50.0 * 1e3:.0f} mm, |e| = {e.norm().item() * 1e3:.2f} mm "
          f"(expected {cfg.max_offset * 1e3:.0f} mm)\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
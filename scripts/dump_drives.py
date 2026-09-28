import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(args.usd_path)
for prim in stage.Traverse():
    for token in ("angular", "linear"):
        drive = UsdPhysics.DriveAPI.Get(prim, token)
        if drive:
            print(f"{prim.GetName():24s} {token:8s} "
                  f"stiffness={drive.GetStiffnessAttr().Get()} "
                  f"damping={drive.GetDampingAttr().Get()} "
                  f"maxForce={drive.GetMaxForceAttr().Get()}")

app.close()
import os; os._exit(0)
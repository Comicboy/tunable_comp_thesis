import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(args.usd_path)
print("default prim:", stage.GetDefaultPrim().GetPath())
print("meters per unit:", __import__("pxr").UsdGeom.GetStageMetersPerUnit(stage))
print()

for p in stage.Traverse():
    flags = []
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        flags.append("ARTICULATION_ROOT")
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        flags.append("rigid")
    if p.IsA(UsdPhysics.Joint):
        flags.append("JOINT")
    print(f"{str(p.GetPath()):65s} {p.GetTypeName():18s} {' '.join(flags)}")

app.close()
import os; os._exit(0)
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdUtils

stage = Usd.Stage.Open(args.usd_path)
print("default prim:", stage.GetDefaultPrim().GetPath())

layers, assets, unresolved = UsdUtils.ComputeAllDependencies(args.usd_path)
print("\nlayers:")
for l in layers:
    print("  ", l.identifier)
print("assets:")
for a in assets:
    print("  ", a)
print("UNRESOLVED (missing):")
for u in unresolved:
    print("  ", u)

app.close()
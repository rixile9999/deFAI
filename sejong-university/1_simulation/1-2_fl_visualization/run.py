from pathlib import Path
import os, runpy, sys
HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
if not any(a == "--store" or a.startswith("--store=") for a in args):
    args += ["--store", str(HERE / "runs")]
sys.argv = [str(HERE / "scripts/run_monitor.py")] + args
os.chdir(str(HERE))
runpy.run_path(sys.argv[0], run_name="__main__")

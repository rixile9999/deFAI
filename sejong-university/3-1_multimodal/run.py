from pathlib import Path
import os, runpy, sys
HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
if not any(a == "--out" or a.startswith("--out=") for a in args):
    args += ["--out", str(HERE / "results")]
sys.argv = [str(HERE / "scripts/train_multimodal.py")] + args
os.chdir(str(HERE))
runpy.run_path(sys.argv[0], run_name="__main__")

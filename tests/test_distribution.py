"""Guard the actual wheel, rather than an editable checkout that can hide omissions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile


def test_built_wheel_has_canonical_code_and_works_outside_source_tree(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "dist"
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(destination)],
                   cwd=root, check=True, capture_output=True, timeout=120)
    wheel = next(destination.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {"defai/__init__.py", "defai/cli.py", "defai/research/__init__.py"} <= names
        assert not any("__pycache__" in name or name.endswith((".jpg", ".xpt", ".ctx", ".apk")) for name in names)
    environment = tmp_path / "clean environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
                   check=True, capture_output=True, timeout=120)
    program = (
        "import defai,json; from defai.research import create_workspace,list_features; "
        f"p=create_workspace({str(tmp_path / 'extracted workspace')!r}); "
        "assert 'site-packages' in defai.__file__; "
        "print(json.dumps({'version':defai.__version__,'features':len(list_features()),'workspace':str(p)}))"
    )
    result = subprocess.run([str(python), "-I", "-c", program], cwd=tmp_path, check=True,
                            text=True, capture_output=True, timeout=60)
    evidence = json.loads(result.stdout)
    assert evidence["version"] == "0.2.0" and evidence["features"] >= 115

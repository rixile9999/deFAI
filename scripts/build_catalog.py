"""Regenerate the exhaustive, static research catalog without importing studies."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {"01": "neonatal", "02": "policy", "03": "generation", "04": "hba1c"}


def build():
    resources, modules = [], []
    candidates = [*ROOT.glob("disease_research/**/*"), *ROOT.glob("scripts/*.py"),
                  *ROOT.glob("defai/**/*.py"), *ROOT.glob("docs/*.md"), ROOT / "README.md",
                  ROOT / ".gitattributes", ROOT / ".gitignore", ROOT / "pyproject.toml", ROOT / "LICENSE", ROOT / "hatch_build.py"]
    for path in sorted(set(candidates), key=lambda p: p.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or any(p in {"__pycache__", ".pytest_cache", "_bundle", "build", "bin", ".gradle", ".idea"} for p in path.parts):
            continue
        if path.suffix in {".pyc", ".log", ".pt", ".pth", ".npz", ".ctx"}:
            continue
        if relative.startswith("disease_research/"):
            allowed = {".py", ".kt", ".kts", ".java", ".xml", ".properties", ".pro", ".sh", ".bat", ".ps1", ".md", ".txt"}
            reference = "/references/" in relative and path.suffix == ".csv"
            wrapper = relative.endswith("/gradle/wrapper/gradle-wrapper.jar")
            if path.suffix not in allowed and not reference and not wrapper and path.name not in {"gradlew", "source_manifest.json", ".gitignore"}:
                continue
            if wrapper and hashlib.sha256(path.read_bytes()).hexdigest() != "7d3a4ac4de1c32b59bc6a4eb8ecb8e612ccd0cf1ae1e99f66902da64df296172":
                raise ValueError("Gradle wrapper does not match the verified upstream artifact")
            if re.search(r"gv[-_ ]?he|geometric.volume.he", relative, re.I):
                raise ValueError(f"Unpublished resource is outside the package scope: {relative}")
        resources.append(relative)
        if not relative.startswith("disease_research/") or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        parts = path.relative_to(ROOT / "disease_research").parts
        domain = TOPICS[parts[0][:2]]
        slug = ".".join(re.sub(r"[^a-z0-9_]+", "_", p.lower()).strip("_") for p in (*parts[1:-1], path.stem))
        arguments = [a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                     for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        main_guard = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) for n in tree.body)
        role = "test" if path.name.startswith("test_") or "tests" in parts else "script" if main_guard or arguments else "library"
        if not main_guard and role == "library" and any(isinstance(n, (ast.For, ast.With)) or
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body):
            role = "script"
        modules.append({"id": domain + "." + slug, "domain": domain, "path": relative,
                        "role": role, "arguments": arguments, "has_main_guard": main_guard,
                        "description": (ast.get_docstring(tree) or path.stem).splitlines()[0],
                        "symbols": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]})
    find = lambda suffix: next(r for r in resources if r.endswith(suffix))
    apis = {"generation": find("fl_pipeline/federated_burr.py"), "paillier": find("fl_pipeline/paillier.py"),
            "policy": find("core/policy_stats.py"), "hba1c": find("fl_comparison/learning.py"),
            "cohort": find("fl_comparison/cohort.py"), "neonatal_data": find("GAN_research/prepare_data.py"),
            "neonatal_model": find("NeoJaundice/src/model.py"),
            "neonatal_aggregation": find("NeoJaundice/src/federated.py"),
            "gan": find("GAN_research/gan_models.py"), "neonatal_algorithms": find("GAN_research/federated_algorithms.py")}
    catalog = {"schema": 1, "resources": resources, "api_modules": apis, "modules": modules,
               "resource_hashes": {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest() for r in resources}}
    (ROOT / "defai/research/catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"resources": len(resources), "modules": len(modules)}))


if __name__ == "__main__":
    build()

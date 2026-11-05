"""Build the pinned Java UDT library and adapter with a local JDK (no Maven)."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def build(output):
    from defai.research.resources import source_path

    root = source_path("java/udt/UPSTREAM.md").parent
    compiler = shutil.which("javac")
    if not compiler:
        raise ValueError("JDK 17 or later is required on PATH")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        [
            *root.glob("vendor/udt-java/src/main/java/**/*.java"),
            *root.glob("src/main/java/**/*.java"),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="defai-java-") as temporary:
        classes = Path(temporary) / "classes"
        classes.mkdir()
        result = subprocess.run(
            [
                compiler,
                "--release",
                "8",
                "-encoding",
                "UTF-8",
                "-d",
                str(classes),
                *map(str, sources),
            ],
            check=False,
        )
        if result.returncode:
            raise ValueError("Java compilation failed; inspect compiler output")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:

            def add(name, data):
                archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data)

            add(
                "META-INF/MANIFEST.MF",
                b"Manifest-Version: 1.0\r\nMain-Class: org.defai.udt.Main\r\n\r\n",
            )
            add(
                "META-INF/udt-java-LICENSE",
                (root / "vendor/udt-java/LICENSE").read_bytes(),
            )
            add("META-INF/UDT-UPSTREAM.md", (root / "UPSTREAM.md").read_bytes())
            for path in sorted(classes.rglob("*.class")):
                add(path.relative_to(classes).as_posix(), path.read_bytes())
    return str(output)

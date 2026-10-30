"""Build the maintained Android app outside non-ASCII research-directory paths."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from defai.research.resources import catalog, source_path


def build_android(output, sdk, *, java_home=None, timeout=600):
    """Build a debug APK from bundled sources; SDK/JDK are supplied by the user."""
    output, sdk = Path(output).resolve(), Path(sdk).resolve()
    if output.exists():
        raise FileExistsError(f"APK output already exists: {output}")
    if not sdk.is_dir():
        raise ValueError("Android SDK directory does not exist")
    if timeout <= 0:
        raise ValueError("Build timeout must be positive")
    resources = catalog()["resources"]
    prefix = next(p.rsplit("/settings.gradle.kts", 1)[0] for p in resources if p.endswith("/defai/settings.gradle.kts")) + "/"
    env = os.environ.copy()
    env["ANDROID_HOME"] = str(sdk)
    if java_home:
        env["JAVA_HOME"] = str(Path(java_home).resolve())
    with tempfile.TemporaryDirectory(prefix="defai-android-") as directory:
        project = Path(directory)
        if not str(project).isascii():
            raise ValueError("Android requires an ASCII build path on Windows; set TEMP to an ASCII directory")
        for relative in resources:
            if not relative.startswith(prefix):
                continue
            local = relative[len(prefix):]
            # The Gradle application module is app/. Root src/ files are older prototypes.
            if local.startswith("src/"):
                continue
            source = source_path(relative)
            if hashlib.sha256(source.read_bytes()).hexdigest() != catalog()["resource_hashes"][relative]:
                raise ValueError(f"Android source integrity mismatch: {relative}")
            target = project / local
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        launcher = ["cmd.exe", "/d", "/c", "gradlew.bat"] if os.name == "nt" else ["sh", "gradlew"]
        subprocess.run([*launcher, "--no-daemon", "--console=plain", ":app:assembleDebug"],
                       cwd=project, env=env, timeout=timeout, check=True)
        apk = project / "app/build/outputs/apk/debug/app-debug.apk"
        if not apk.is_file():
            raise ValueError("Gradle completed without producing the expected APK")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            staged = Path(stream.name)
            with apk.open("rb") as source:
                shutil.copyfileobj(source, stream)
        try:
            os.link(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    return {"apk": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}

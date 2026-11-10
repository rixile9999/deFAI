import threading
import time
from typing import Dict, List

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class ResourceSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: List[Dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process() if psutil else None

    def start(self) -> "ResourceSampler":
        if self._proc is None:
            return self
        self._proc.cpu_percent(None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = {"t": time.time(), "cpu": self._proc.cpu_percent(None),
                     "mem": self._proc.memory_info().rss / 2 ** 20,
                     "sys_cpu": psutil.cpu_percent(None), "sys_mem": psutil.virtual_memory().percent}
                with self._lock:
                    self._samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def drain(self) -> Dict[str, float]:
        with self._lock:
            s, self._samples = self._samples, []
        if not s:
            return {}
        out = {}
        for k in ("cpu", "mem", "sys_cpu", "sys_mem"):
            v = [x[k] for x in s]
            out[k + "_min"], out[k + "_mean"], out[k + "_max"] = min(v), sum(v) / len(v), max(v)
        out["n"] = len(s)
        return out

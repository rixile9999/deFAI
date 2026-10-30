"""Record reproducibility metadata without allocating model tensors."""
from datetime import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
import psutil
from cohort import ROOT

gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                    '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
cpu=subprocess.run(['powershell','-NoProfile','-Command',
                    '(Get-CimInstance Win32_Processor).Name'],capture_output=True,text=True,check=True).stdout.strip()
records=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'experiments/fl_comparison/results/main').glob('seed_*/*/validation.json')]
output={'recorded_at_kst':datetime.now().isoformat(),'python':sys.version,
    'python_executable':sys.executable,'platform':platform.platform(),'cpu':cpu,
    'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),
    'system_ram_bytes':psutil.virtual_memory().total,'gpu_name_driver_memory':gpu,
    'packages':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','torch','psutil','matplotlib')},
    'sharing_note':'Other GPU jobs overlapped training, one exited independently around 10:17 and another started later. No other user process was terminated.',
    'cpu_probe_note':'CPU worker uses saved exact CUDA-generator initial arrays. FedAvg full-run equivalence probe differed by about 4e-8 percentage points. Training device is recorded for each run; main test inference uses CUDA.',
    'successful_training_counts_by_device':dict(Counter(r['device'] for r in records)),
    'gpu_persistent_state':'Client SCAFFOLD and BN states on CPU; only selected clients transferred to GPU.',
    'per_process_cuda_allocator_fraction':.02}
path=ROOT/'experiments/fl_comparison/results/main/runtime_environment.json'
path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))

from pathlib import Path
__version__ = "3.2.0"
_root = Path(__file__).resolve().parents[3]
for _name in ("1_simulation/1-2_fl_visualization", "1_simulation/1-4_distributed_comparison", "3_heterogeneous_data/3-1_multimodal"):
    _part = _root / _name / "src" / "defai_addons"
    if _part.is_dir():
        __path__.append(str(_part))

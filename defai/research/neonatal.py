"""Neonatal image/metadata models and patient-weighted aggregation."""
from defai.research.resources import load_api
from pathlib import Path

MultiModalNet = load_api("neonatal_model").MultiModalNet
average_states = load_api("neonatal_aggregation").average_states
Generator = load_api("gan").Generator
Discriminator = load_api("gan").Discriminator
proximal_penalty = load_api("neonatal_algorithms").proximal_penalty
scaffold_correct = load_api("neonatal_algorithms").scaffold_correct
scaffold_control = load_api("neonatal_algorithms").scaffold_control


def prepare_dataset(source, output, *, seed=42):
    """Prepare a patient-disjoint dataset in a user-selected output directory."""
    return load_api("neonatal_data").prepare(Path(output), seed, Path(source))

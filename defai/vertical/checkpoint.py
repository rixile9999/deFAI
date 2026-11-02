"""Save model/optimizer state without serializing participant features or labels."""
import copy
import os
from pathlib import Path
import tempfile
import torch


def save_checkpoint(trainer, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if any(p._pending is not None for p in trainer.parties):
        raise RuntimeError('Complete outstanding batches before checkpointing')
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'schema': 1, 'names': [p.name for p in trainer.parties], 'task': trainer.task,
             'bottoms': [p.model.state_dict() for p in trainer.parties],
             'bottom_optimizers': [p.optimizer.state_dict() if p.optimizer else None for p in trainer.parties],
             'top': trainer.top.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'steps': trainer.steps,
             'cpu_rng': torch.random.get_rng_state()}
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication on supported local filesystems.
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(trainer, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('schema') != 1 or state.get('names') != [p.name for p in trainer.parties] or state.get('task') != trainer.task:
        raise ValueError('Checkpoint schema, task or participant names differ')
    if any(p._pending is not None for p in trainer.parties):
        raise RuntimeError('Cannot restore into a pending batch')
    if len(state['bottoms']) != len(trainer.parties) or len(state['bottom_optimizers']) != len(trainer.parties):
        raise ValueError('Checkpoint party count differs')
    # Validate the entire state on copies before touching the live trainer.
    model_copies = [copy.deepcopy(p.model) for p in trainer.parties] + [copy.deepcopy(trainer.top)]
    for model, saved in zip(model_copies, state['bottoms']+[state['top']]):
        model.load_state_dict(saved, strict=True)
        if any(not torch.isfinite(value).all() for value in saved.values()):
            raise ValueError('Checkpoint contains non-finite model state')
    for party, saved in zip(trainer.parties, state['bottom_optimizers']):
        if (party.optimizer is None) != (saved is None):
            raise ValueError('Checkpoint trainable-party configuration differs')
        if saved is not None:
            copy.deepcopy(party.optimizer).load_state_dict(saved)
    copy.deepcopy(trainer.optimizer).load_state_dict(state['optimizer'])
    if not isinstance(state['steps'], int) or state['steps'] < 0:
        raise ValueError('Invalid checkpoint step count')
    torch.Generator().set_state(state['cpu_rng'])
    for party, saved, optimizer in zip(trainer.parties, state['bottoms'], state['bottom_optimizers']):
        party.model.load_state_dict(saved)
        if party.optimizer:
            party.optimizer.load_state_dict(optimizer)
    trainer.top.load_state_dict(state['top'])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.steps = state['steps']
    torch.random.set_rng_state(state['cpu_rng'])

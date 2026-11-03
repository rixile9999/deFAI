"""Save model/optimizer state without serializing participant features or labels."""
import copy
import math
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
    if hasattr(trainer, 'q'):
        state['block_config'] = {'q': trainer.q, 'mode': trainer.mode, 'owner': trainer.owner}
        state['local_steps'] = trainer.local_steps
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
    block = {'q': trainer.q, 'mode': trainer.mode, 'owner': trainer.owner} if hasattr(trainer, 'q') else None
    if state.get('block_config') != block:
        raise ValueError('Checkpoint block-update configuration differs')
    if block and (not isinstance(state.get('local_steps'), int) or state['local_steps'] < 0):
        raise ValueError('Invalid local block-step count')
    if len(state['bottoms']) != len(trainer.parties) or len(state['bottom_optimizers']) != len(trainer.parties):
        raise ValueError('Checkpoint party count differs')
    # Validate the entire state on copies before touching the live trainer.
    for saved in [state['optimizer'], *state['bottom_optimizers']]:
        if saved is not None and any(not math.isfinite(group['lr']) or group['lr'] <= 0 for group in saved['param_groups']):
            raise ValueError('Checkpoint contains an invalid optimizer learning rate')
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
    if block:
        trainer.local_steps = state['local_steps']
    torch.random.set_rng_state(state['cpu_rng'])

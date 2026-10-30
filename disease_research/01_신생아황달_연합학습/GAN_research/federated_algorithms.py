"""FedProx objective, SCAFFOLD option II, and local BN state handling."""
from collections import OrderedDict
import torch


def bn_keys(model):
    return {f"{name}.{key}" for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key in module.state_dict()}


def train_mode(model, local_bn=False):
    model.train()
    if local_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()


def proximal_penalty(model, anchor):
    return sum((p-anchor[name].to(p.device)).square().sum() for name, p in model.named_parameters())/2


@torch.no_grad()
def scaffold_correct(model, server_control, client_control):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.add_(server_control[name].to(parameter.device)-client_control[name].to(parameter.device))


def scaffold_control(old_client, old_server, anchor, local, steps, lr):
    if steps <= 0 or lr <= 0:
        raise ValueError("SCAFFOLD requires positive actual SGD step count and learning rate")
    return OrderedDict((name, old_client[name]-old_server[name]+(anchor[name].cpu()-local[name].cpu())/(steps*lr))
                       for name in old_client)


def personalized_state(shared, local_bn):
    return OrderedDict((key, local_bn.get(key, value)) for key, value in shared.items())

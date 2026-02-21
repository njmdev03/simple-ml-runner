from typing import Callable, Dict, Any
import torch.nn as nn

_model_builders: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {}


def register_model_builder(name: str, builder: Callable[[Dict[str, Any]], nn.Module]):
    _model_builders[name.lower()] = builder


def get_model_builder(name: str):
    return _model_builders.get(name.lower())


def mlp_builder(params: Dict[str, Any]) -> nn.Module:
    # Expected params: input_dim, hidden_layers (list), output_dim, activation
    input_dim = int(params.get('input_dim', 128))
    hidden = params.get('hidden_layers', []) or []
    hidden = [int(h) for h in hidden]
    output_dim = int(params.get('output_dim', 10))
    activation = params.get('activation', 'relu')

    act_cls = nn.ReLU
    if isinstance(activation, str):
        if activation.lower() in ('relu', 'relu6'):
            act_cls = nn.ReLU
        elif activation.lower() in ('tanh',):
            act_cls = nn.Tanh
        elif activation.lower() in ('sigmoid',):
            act_cls = nn.Sigmoid

    layers = []
    in_dim = input_dim
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(act_cls())
        in_dim = h

    layers.append(nn.Linear(in_dim, output_dim))

    return nn.Sequential(*layers)


# register default builders
register_model_builder('mlp', mlp_builder)
register_model_builder('MLP', mlp_builder)

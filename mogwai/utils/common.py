from typing import Callable
import functools
import numpy as np
import torch


def coerce_numpy(func: Callable) -> Callable:
    """ Allows user to pass numpy arguments to a torch function and auto-converts back to
    numpy at the end.
    """
    @functools.wraps(func)
    def make_torch_args(*args, **kwargs):
        is_numpy = False
        update_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                is_numpy = True
            update_args.append(arg)
        update_kwargs = {}
        for kw, arg in kwargs.items():
            if isinstance(args, np.ndarray):
                arg = torch.from_numpy(arg)
                is_numpy = True
            update_kwargs[kw] = arg

        output = func(*update_args, **update_kwargs)

        if is_numpy:
            output = recursive_make_numpy(output)

        return output

    return make_torch_args


def recursive_make_torch(item):
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    elif isinstance(item, (tuple, list)):
        return type(item)(recursive_make_torch(el) for el in item)
    elif isinstance(item, dict):
        return {kw: recursive_make_torch(arg) for kw, arg in item.items()}
    else:
        return item


def recursive_make_numpy(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    elif isinstance(item, (tuple, list)):
        return type(item)(recursive_make_numpy(el) for el in item)
    elif isinstance(item, dict):
        return {kw: recursive_make_numpy(arg) for kw, arg in item.items()}
    else:
        return item

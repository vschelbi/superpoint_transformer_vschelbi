import torch


__all__ = ['print_memory_size']


def print_memory_size(a):
    assert isinstance(a, torch.Tensor)
    memory = a.element_size() * a.nelement()
    if memory > 1024 * 1024 * 1024:
        print(f'Memory: {memory / (1024 * 1024 * 1024):0.3f} Gb')
        return
    if memory > 1024 * 1024:
        print(f'Memory: {memory / (1024 * 1024):0.3f} Mb')
        return
    if memory > 1024:
        print(f'Memory: {memory / 1024:0.3f} Kb')
        return
    print(f'Memory: {memory:0.3f} bytes')

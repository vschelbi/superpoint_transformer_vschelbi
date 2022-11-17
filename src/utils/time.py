import torch
from time import time


__all__ = ['timer']


def timer(f, *args, text='', text_size=64, **kwargs):
    torch.cuda.synchronize()
    start = time()
    out = f(*args, **kwargs)
    torch.cuda.synchronize()
    padding = '.' * (text_size - len(text))
    print(f'{text}{padding}: {time() - start:0.3f}s')
    return out

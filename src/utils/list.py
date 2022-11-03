__all__ = ['listify']


def listify(obj):
    if obj is None or isinstance(obj, str):
        return obj
    if not hasattr(obj, '__len__'):
        return obj
    if hasattr(obj, 'dim') and obj.dim() == 0:
        return obj
    if len(obj) == 0:
        return obj
    return [listify(x) for x in obj]

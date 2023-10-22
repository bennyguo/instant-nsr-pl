datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import blender, colmap, dtu
try:
    # customized data
    import R3DParser
    from . import hmvs
except ImportError:
    pass

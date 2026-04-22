def njit(*args, **kwargs):
    def decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

def jit(*args, **kwargs):
    return njit(*args, **kwargs)

prange = range

def vectorize(*args, **kwargs):
    return njit(*args, **kwargs)

def guvectorize(*args, **kwargs):
    return njit(*args, **kwargs)

int32 = int
int64 = int
float32 = float
float64 = float
bool_ = bool

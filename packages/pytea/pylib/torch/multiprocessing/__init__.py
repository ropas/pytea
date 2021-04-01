from multiprocessing import *

# TODO: return ProcessContext if join is False
def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    fn(0, *args)

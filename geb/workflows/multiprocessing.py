import signal
from functools import wraps
from typing import Any, Callable


def handle_ctrl_c(func) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler)  # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt

    return wrapper


def pool_ctrl_c_handler(*args: Any, **kwargs: Any) -> None:
    """Handle Ctrl-C in multiprocessing pool workers.

    Sets a global flag to indicate that Ctrl-C has been pressed which can be checked by worker processes.
    """
    global ctrl_c_entered
    ctrl_c_entered = True


def init_pool(
    manager_current_gpu_use_count, manager_lock, gpus, models_per_gpu
) -> None:
    # set global variable for each process in the pool:
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

    global lock
    global current_gpu_use_count
    global n_gpu_spots
    n_gpu_spots = gpus * models_per_gpu
    lock = manager_lock
    current_gpu_use_count = manager_current_gpu_use_count

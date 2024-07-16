import torch

def test(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            func(*args, **kwargs)
            print(f"\033[92mTest {func_name} passed\033[0m")  
        except AssertionError as e: 
            print(f"\033[91mTest {func_name} failed: {e}\033[0m")  
    return wrapper

def assert_all_equal(t1: torch.Tensor, t2: torch.Tensor):
    assert torch.all(t1 == t2), f"{t1} != {t2}"

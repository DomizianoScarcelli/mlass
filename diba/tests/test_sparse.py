import torch
from diba.diba.sparse_utils import sparse_permute, sparse_expand,sparse_expand_as, sparse_elementwise_div, sparse_normalize



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

@test
def test_sparse_permute():
    t = torch.randn((256,256,256))
    t_sparse = t.to_sparse_coo()
    t_permuted = t.permute(1,2,0)
    t_sparse_permuted = sparse_permute(t_sparse, (1,2,0)).to_dense()
    assert_all_equal(t_permuted, t_sparse_permuted)

@test
def test_sparse_expand_2():
    t = torch.randn((1,5))
    t_sparse = t.to_sparse_coo()
    expanded = t.expand((5,5))
    expanded_sparse = sparse_expand(t_sparse, (5,5)).to_dense()
    assert_all_equal(expanded, expanded_sparse)

@test
def test_sparse_expand_3():
    t = torch.randn((1,5,1))
    t_sparse = t.to_sparse_coo()
    expanded = t.expand((5,5,5))
    expanded_sparse = sparse_expand(t_sparse, (5,5,5)).to_dense()
    assert_all_equal(expanded, expanded_sparse)

@test
def test_sparse_expand_4():
    t = torch.randn((1,5,1,5))
    t_sparse = t.to_sparse_coo()
    expanded = t.expand((5,5,5,5))
    expanded_sparse = sparse_expand(t_sparse, (5,5,5,5)).to_dense()
    assert_all_equal(expanded, expanded_sparse)

@test
def test_sparse_expand_4_high():
    K = 2048
    t = torch.randn((1,K,1,K))
    t_sparse = t.to_sparse_coo()
    assert(sparse_expand(t_sparse, (K,K,K,K)))
                                                                                                           
@test
def test_sparse_expand_as():
    t = torch.randn((1,5,1))
    t_sparse = t.to_sparse_coo()
    expanded = t.expand((5,5,5))
    expanded_sparse = sparse_expand_as(t_sparse, expanded).to_dense()
    assert_all_equal(expanded, expanded_sparse)

@test
def sparse_elementwise_div():
    pass

@test
def sparse_normalize():
    pass

if __name__ == "__main__":
    torch.manual_seed(0)
    test_sparse_permute()
    test_sparse_expand_2()
    test_sparse_expand_3()
    test_sparse_expand_4()
    test_sparse_expand_4_high()
    test_sparse_expand_as()

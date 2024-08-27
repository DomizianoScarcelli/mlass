import torch
from diba.diba.sparse_utils import sparse_permute, sparse_expand,sparse_expand_as, sparse_elementwise_div, sparse_normalize
import sparse
from diba.tests.utils import test, assert_all_equal


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
def test_sparse_normalize_rank_3():
    K = 5
    NORM_DIM = 1
    t = torch.randn((3,K,K))
    t_sparse = t.to_sparse_coo()
    dense_normalizer = t.sum(dim=NORM_DIM).unsqueeze(NORM_DIM)
    print("Dense normalizer: ", dense_normalizer, dense_normalizer.shape)
    print("Dense values: ", t)
    t_normalized = t / dense_normalizer
    t_sparse_normalized = sparse_normalize(t_sparse, dim=NORM_DIM)
    assert torch.allclose(t_normalized, t_sparse_normalized.to_dense()), f"""
    t_normalized: {t_normalized, t_normalized.shape}
    t_sparse_normalized: {t_sparse_normalized.to_dense(), t_sparse_normalized.shape}
    """

@test
def test_sparse_normalize_rank_4():
    K = 3
    NORM_DIM = 1
    t = torch.randn((2,K,K,K))
    t_sparse = t.to_sparse_coo()
    dense_normalizer = t.sum(dim=NORM_DIM).unsqueeze(NORM_DIM)
    print("Dense normalizer: ", dense_normalizer, dense_normalizer.shape)
    # print("Dense values: ", t)
    t_normalized = t / dense_normalizer 
    t_sparse_normalized = sparse_normalize(t_sparse, dim=NORM_DIM)
    # assert torch.allclose(t_normalized, t_sparse_normalized.to_dense()), f"""
    # t_normalized: {t_normalized, t_normalized.shape}
    # t_sparse_normalized: {t_sparse_normalized.to_dense(), t_sparse_normalized.shape}
    # """
    pass

@test
def test_sparse_sum():
    K = 5
    NORM_DIM = 1
    t = torch.randn((4,K,K,K))
    t_sparse = t.to_sparse_coo()
    t_sum = t.sum(dim=NORM_DIM)
    t_sparse_sum = torch.sparse.sum(t_sparse, dim=NORM_DIM).to_dense()
    assert torch.allclose(t_sum, t_sparse_sum), f"""
    t_sum: {t_sum}
    t_sparse_sum: {t_sparse_sum}
    """

if __name__ == "__main__":
    torch.manual_seed(0)
    # test_sparse_permute()
    # test_sparse_expand_2()
    # test_sparse_expand_3()
    # test_sparse_expand_4()
    # test_sparse_expand_4_high()
    # test_sparse_expand_as()
    # test_sparse_sum()
    test_sparse_normalize_rank_3()
    test_sparse_normalize_rank_4()

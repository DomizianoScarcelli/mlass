import time
import torch

from diba.diba.sparse_utils import slice_along_index
from lass_mnist.lass.diba_interaces import SparseLikelihood

torch.set_printoptions(precision=2, sci_mode=False)
num_tokens = 3
num_sources = 2


def get_index(num_sources: int, message_to: int):
    """
    (-1, 1, 1) if message to m, (1, -1, 1) if message to z1, (1, 1, -1) if message to z2
    """
    indexing = []
    for i in range(num_sources):
        if i == message_to:
            indexing.append(-1)
        else:
            indexing.append(1)
    return indexing

def exclude_row(tensor: torch.Tensor, row: int):
    """
    Exclude a row from a tensor
    """
    return torch.cat((tensor[:row], tensor[row+1:]))

################
# Test 1: Likelihood to mixture message update
################


def test_1():
    msg_z1_fl1 = torch.log(torch.softmax(
        torch.randn(size=(num_tokens,)), dim=0))
    msg_z2_fl2 = torch.log(torch.softmax(
        torch.randn(size=(num_tokens,)), dim=0))

    fl_value = torch.softmax(torch.randn(
        size=(num_tokens, num_tokens, num_tokens)), dim=0)

    sum_messages = torch.exp(msg_z1_fl1 + msg_z2_fl2)

    product_factor_sum_messages = torch.log(
        fl_value * sum_messages.view(tuple(get_index(num_sources=num_sources + 1, message_to=0))))

    print(f"msg_z1_fl1 is {msg_z1_fl1} with shape {msg_z1_fl1.shape}")
    print(f"msg_z2_fl2 is {msg_z2_fl2} with shape {msg_z2_fl2.shape}")
    print(f"fl_value is {fl_value} with shape {fl_value.shape}")
    print(f"sum_messages is {sum_messages} with shape {sum_messages.shape}")
    print(
        f"product_factor_sum_messages is {product_factor_sum_messages} with shape {product_factor_sum_messages.shape}")


################
# Test 2: Batched likelihood to variable message update
################
def test_2():
    msg_to_fl = torch.log(torch.softmax(
        torch.randn(size=(num_sources + 1, num_tokens)), dim=0))

    fl_value = torch.softmax(torch.randn(
        size=(num_tokens, num_tokens, num_tokens)), dim=0)

    sum_messages = torch.exp(torch.sum(msg_to_fl, dim=0))

    msg_fl_variables = torch.empty(size=(num_sources + 1, *fl_value.shape))

    for i in range(num_sources + 1):
        indexes = get_index(num_sources=num_sources + 1, message_to=i)
        product_factor_sum_messages = torch.log(
            fl_value * sum_messages.view(tuple(indexes)))
        msg_fl_variables[i] = product_factor_sum_messages

    print(f"msg_z_fl1 is {msg_to_fl} with shape {msg_to_fl.shape}")
    print(f"fl_value is {fl_value} with shape {fl_value.shape}")
    print(f"sum_messages is {sum_messages} with shape {sum_messages.shape}")
    print(
        f"msg_fl_variables is {msg_fl_variables} with shape {msg_fl_variables.shape}")

    assert not torch.allclose(msg_fl_variables[0], msg_fl_variables[1]) and not torch.allclose(
        msg_fl_variables[0], msg_fl_variables[2]) and not torch.allclose(msg_fl_variables[1], msg_fl_variables[2])

################
# Test 3: Batched variable to prior message update
################


def test_3():
    # Incoming messages to z.
    messages_to_z = torch.log(torch.softmax(
        torch.randn(size=(num_sources + 1, num_tokens, num_tokens, num_tokens)), dim=0))
    print(f"messages_to_z is {messages_to_z} with shape {messages_to_z.shape}")

    exclude_z1 = exclude_row(messages_to_z, row=1)

    sum_messages = torch.sum(exclude_z1, dim=0)

    print(
        f"Incoming sum messages to z is {sum_messages} with shape {sum_messages.shape}")
    pr_shape = (num_sources, num_tokens)
    pr_value = torch.softmax(torch.randn(
        size=(pr_shape)), dim=1)

    z_shape = (num_sources + 1, num_tokens)
    message_z_to_pr = sum_messages

    print(
        f"Message z to pr is {message_z_to_pr} with shape {message_z_to_pr.shape}")


def test_4():
    """
    Test 4: Compute marginal posterior
    """
    message_pr_to_z = torch.log(torch.softmax(
        torch.randn(size=(2, num_tokens)), dim=0))
    message_fl_to_z = torch.log(torch.softmax(
        torch.randn(size=(num_tokens, num_tokens, num_tokens)), dim=0))

    sum_messages = message_fl_to_z + message_pr_to_z[0] + message_pr_to_z[1]

    print(f"Sum messages is {sum_messages} with shape {sum_messages.shape}")


def test_sparse_normalization():
    dummy_tensor = torch.zeros(size=(3, 3, 3, 3))
    random_indices = torch.tensor([[1, 2, 0, 2],
                                   [1, 2, 0, 0],
                                   [1, 2, 1, 1],
                                   [2, 0, 2, 0],
                                   [1, 2, 0, 0],
                                   [2, 0, 2, 0],
                                   [2, 1, 2, 1],
                                   [0, 1, 2, 2],
                                   [2, 1, 2, 0],
                                   [0, 1, 2, 2]])

    print(random_indices)
    dummy_tensor[random_indices[0], random_indices[1],
                 random_indices[2], random_indices[3]] = 1

    dense_normalization = torch.sum(dummy_tensor, dim=-1)
    dense_mask = dense_normalization != 0.0
    dummy_tensor[dense_mask] = dummy_tensor[dense_mask] / \
        dense_normalization[dense_mask].unsqueeze(-1)

    print(f"Dense tensor normalized is {dummy_tensor}")
    print(
        f"Dense tensor normalized converted to sparse is {dummy_tensor.to_sparse()}")

    sparse_tensor = dummy_tensor.to_sparse()
    sparse_normalization = torch.sparse.sum(sparse_tensor, dim=-1)

    print(f"Sparse normalization is {sparse_normalization}")

    sparse_tensor = sparse_tensor.coalesce()
    sparse_normalization = sparse_normalization.coalesce()
    normalized_indices = []
    normalized_values = []
    for i, index in enumerate(sparse_normalization.indices()):
        normalized_indices.append(index)
        normalized_value = sparse_tensor[index[0], index[1],
                                         index[2]] / sparse_normalization.values()[i]

        print(f"Normalized value for index {index} is {normalized_value}")
        normalized_values.append(normalized_value)

    print(f'Normalized indices are {normalized_indices}')
    print(f'Normalized values are {normalized_values}')

    normalized_sparse_tensor = torch.sparse_coo_tensor(
        torch.cat(normalized_indices),
        torch.cat(normalized_values),
        sparse_tensor.shape,
    )

    assert normalized_sparse_tensor == dummy_tensor.to_sparse()


def test_sparse_slicing():
    dummy_tensor = torch.zeros(size=(3, 3, 3, 3))
    random_indices = torch.tensor([[1, 2, 0, 2],
                                   [1, 2, 0, 0],
                                   [1, 2, 1, 1],
                                   [2, 0, 2, 0],
                                   [1, 2, 0, 0],
                                   [2, 0, 2, 0],
                                   [2, 1, 2, 1],
                                   [0, 1, 2, 2],
                                   [2, 1, 2, 0],
                                   [0, 1, 2, 2]])

    print(random_indices)
    dummy_tensor[random_indices[0], random_indices[1],
                 random_indices[2], random_indices[3]] = 1

    slice_index = 2
    sliced_tensor = dummy_tensor[:, :, slice_index]
    print(f"Sliced tensor is {sliced_tensor}")

    sparse_tensor = dummy_tensor.to_sparse()
    print(f"Sparse tensor is {sparse_tensor}")

    sliced_sparse_tensor = slice_along_index(
        sparse_tensor, slice_index
    )

    print(f"Sparse tensor is {sparse_tensor}")

    assert torch.allclose(sliced_tensor, sliced_sparse_tensor.to_dense()), f"""
    sliced_tensor is {sliced_tensor}
    sliced_sparse_tensor to dense is {sliced_sparse_tensor.to_dense()}
    """


def test_save_and_load_sparse():
    dummy_tensor = torch.zeros(size=(3, 3, 3, 3))
    random_indices = torch.tensor([[1, 2, 0, 2],
                                   [1, 2, 0, 0],
                                   [1, 2, 1, 1],
                                   [2, 0, 2, 0],
                                   [1, 2, 0, 0],
                                   [2, 0, 2, 0],
                                   [2, 1, 2, 1],
                                   [0, 1, 2, 2],
                                   [2, 1, 2, 0],
                                   [0, 1, 2, 2]])

    print(random_indices)
    dummy_tensor[random_indices[0], random_indices[1],
                 random_indices[2], random_indices[3]] = 1

    torch.save(dummy_tensor.to_sparse(), "dummy.pt")

    sparse_tensor: torch.Tensor = torch.load("dummy.pt")

    print(f"Loaded sparse tensor is {sparse_tensor}")


def test_sparse_likelihood_normalization():
    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/256-sigmoid-big.pt"

    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums: torch.Tensor = torch.load(f, map_location=torch.device('cpu'))

    likelihood = SparseLikelihood(sums=sums.to_sparse_coo())


def timeit(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start} seconds")
        return result
    return wrapper


@timeit
def test_sum_complexity():
    import time
    dummy_tensor = torch.randn(size=(150, 150, 150, 150))
    # start = time.time()
    # torch.sum(dummy_tensor, dim=(0, 1, 2))
    # end = time.time()
    # print(f"Sum over batched 3 dims took {end - start} seconds")

    start = time.time()
    summed = torch.sum(dummy_tensor, dim=2)
    summed = torch.sum(summed, dim=1)
    summed = torch.sum(summed, dim=0)
    end = time.time()
    print(f"Sum over 3 dims took {end - start} seconds")

    start = time.time()
    torch.sum(dummy_tensor, dim=(0, 1))
    end = time.time()
    print(f"Sum over 2 dims took {end - start} seconds")


def test_sparse_indexing():
    dummy_tensor = torch.zeros(size=(3, 3, 3, 3))
    random_indices = torch.tensor([[1, 2, 0, 2],
                                   [1, 2, 0, 0],
                                   [1, 2, 1, 1],
                                   [2, 0, 2, 0],
                                   [1, 2, 0, 0],
                                   [2, 0, 2, 0],
                                   [2, 1, 2, 1],
                                   [0, 1, 2, 2],
                                   [2, 1, 2, 0],
                                   [0, 1, 2, 2]])
    dummy_tensor[random_indices] = 1

    dummy_tensor = dummy_tensor.to_sparse()

    indexed = dummy_tensor[((1,), (2,))]
    print(f"Indexed is {indexed}")


def test_concatenation():
    tens = torch.tensor([])
    other_tens = torch.tensor(0.0).unsqueeze(0)
    other_tens_again = torch.tensor(1.0).unsqueeze(0)

    result = torch.cat((tens, other_tens))
    result = torch.cat((result, other_tens_again))

    print(f"Result of concatenation is {result}")


if __name__ == "__main__":
    test_concatenation()

import torch

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


if __name__ == "__main__":
    test_4()

import torch
from tqdm import tqdm


def slice_along_index(sparse_tensor: torch.Tensor, index: int) -> torch.Tensor:
    sparse_tensor = sparse_tensor.coalesce()
    new_indices = []
    new_values = []

    print(
        f"There are {sparse_tensor.indices().shape[1]} indices out of the total {256 ** 4} dimensions")

    print("Transposing indices")
    transposed_indices = sparse_tensor.indices().t()
    print("Transposing indices done")

    # print("Zipping")
    # zipped = zip(transposed_indices, sparse_tensor.values())
    # print("Zipping done")

    values_dict = {}
    i = 0
    for value in tqdm(sparse_tensor.values(), desc="Creating values dict"):
        values_dict[i] = value
        i += 1

    for i, indices in enumerate(tqdm(transposed_indices, desc="Slicing sparse tensor", total=sparse_tensor.indices().shape[1])):
        if indices[index] == index:
            new_indices.append(
                torch.cat((indices[:index], indices[index + 1:],)))
            new_values.append(values_dict[i])

    sliced_sparse_tensor = torch.sparse_coo_tensor(
        torch.stack(new_indices).t(),
        torch.tensor(new_values),
        sparse_tensor.shape[:-1],
    )
    return sliced_sparse_tensor

import torch


def slice_along_index(sparse_tensor: torch.Tensor, index: int) -> torch.Tensor:
    sparse_tensor = sparse_tensor.coalesce()
    new_indices = []
    new_values = []
    for indices, values in zip(sparse_tensor.indices().t(), sparse_tensor.values()):
        if indices[index] == index:
            new_indices.append(
                torch.cat((indices[:index], indices[index + 1:],)))
            new_values.append(values)

    sliced_sparse_tensor = torch.sparse_coo_tensor(
        torch.stack(new_indices).t(),
        torch.tensor(new_values),
        sparse_tensor.shape[:-1],
    )
    return sliced_sparse_tensor

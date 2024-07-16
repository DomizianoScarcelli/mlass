import torch
from tqdm import tqdm
import sparse
from typing import List

def slice_along_index(sparse_tensor: torch.Tensor, index: int) -> torch.Tensor:
    sparse_tensor = sparse_tensor.coalesce()
    new_indices = []
    new_values = []

    transposed_indices = sparse_tensor.indices().t()

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

def sparse_permute(input, dims):
    #source: https://github.com/pytorch/pytorch/issues/78422
    dims = torch.LongTensor(dims)
    return torch.sparse_coo_tensor(indices=input._indices()[dims], values=input._values(), size=torch.Size(torch.tensor(input.size())[dims]))

def convert_sparse_coo_to_torch_coo(sparse_coo: sparse.COO, device) -> torch.Tensor:
    coords = torch.tensor(
        sparse_coo.coords, device=device, dtype=torch.long)
    data = torch.tensor(
        sparse_coo.data, device=device, dtype=torch.float)
    torch_coo = torch.sparse_coo_tensor(coords, data, size=sparse_coo.shape)
    return torch_coo

def convert_torch_coo_to_sparse_coo(torch_coo: torch.Tensor) -> sparse.COO:
    torch_coo = torch_coo.coalesce()
    sparse_coo = sparse.COO(torch_coo.indices(), torch_coo.values(), torch_coo.shape)
    return sparse_coo


def sparse_expand(t, dims):
    t = t.coalesce()
    original_indices = t.indices()
    original_values = t.values()
    original_shape = t.shape

    # Calculate the number of repeats needed for each dimension
    repeats = [dims[dim] // original_shape[dim] for dim in range(len(dims))]

    # Generate new indices for the expanded tensor
    mesh_grids = torch.meshgrid(*[torch.arange(repeats[dim]) for dim in range(len(repeats))], indexing='ij')
    mesh_grids = [grid.flatten() for grid in mesh_grids]

    expanded_indices_list = []
    for dim in range(len(dims)):
        new_indices_dim = mesh_grids[dim].repeat_interleave(original_indices.size(1))
        expanded_indices_list.append(new_indices_dim)

    expanded_indices = torch.stack(expanded_indices_list, dim=0)
    
    # Repeat the original indices and values to match the new shape
    expanded_original_indices = original_indices.repeat(1, torch.prod(torch.tensor(repeats)).item())
    expanded_indices = expanded_indices + expanded_original_indices

    expanded_values = original_values.repeat(torch.prod(torch.tensor(repeats)).item())

    # Create the new sparse tensor with the expanded indices and values
    expanded_sparse_tensor = torch.sparse_coo_tensor(expanded_indices, expanded_values, torch.Size(dims))

    return expanded_sparse_tensor

def sparse_expand_as(t1: torch.Tensor, t2:torch.Tensor) -> torch.Tensor:
    return sparse_expand(t1, dims=list(t2.shape))

def sparse_elementwise_div(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    assert (t1.shape == t2.shape), f"Shape of t1 and t2 must match: {t1.shape} != {t2.shape})"
    t1 = t1.coalesce()
    t2 = t2.coalesce()
    t2_inverted = torch.sparse_coo_tensor(t2.indices(), 1/t2.values(), t2.shape)
    return t1 * t2_inverted

#def sparse_normalize(t: torch.Tensor, dims) -> torch.Tensor:
#    #TODO: this has to be tested
#    integral = t
#    for dim in dims:
#        integral = torch.sparse.sum(integral, dim=dim).unsqueeze(dim)
#    expanded_integral = sparse_expand_as(integral, t)
#    result = sparse_elementwise_div(t, expanded_integral)
#    return result

def sparse_normalize(t: torch.Tensor, dim:int) -> torch.Tensor:
    t = t.coalesce()
    indices = t.indices()
    values = t.values()
    shape = t.shape

    # Sum along the last dimension
    sums = torch.sparse.sum(t, dim=dim).unsqueeze(dim)

    # Ensure sums has a non-zero value to avoid division by zero
    sums_values = sums.coalesce().values()
    sums_values[sums_values == 0] = 1.0

    # Divide each value in the original tensor by the corresponding sum
    normalized_values = values / sums_values[indices[dim]]

    # Create a new sparse tensor with the updated values
    result = torch.sparse_coo_tensor(indices, normalized_values, size=shape)

    return result

def sparse_normalize_alt(t: sparse.COO, dims) -> torch.Tensor:
    #NOTE: this doesn't work on M1 macos because of numba + sparse
    integrals = t.sum(axis=dims, keepdims=True)
    integrals = sparse.COO(
        integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
    )
    return t / integrals

if __name__ == "__main__":
    pass
    # t1 = torch.randn((1,256,1,1)).to_sparse_coo()
    # t2 = torch.randn((1,256,256,256)).to_sparse_coo()
    # t1_expanded = sparse_expand(t1, dims=list(t2.shape))
    # division = sparse_elementwise_div(t2, t1_expanded)
    # print(t1.shape, t1_expanded.shape, division)

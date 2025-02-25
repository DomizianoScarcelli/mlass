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


def sparse_expand(t1, dims):
    coo = sparse.COO(t1.indices(), t1.values(), t1.shape)
    coo = coo.broadcast_to(dims)
    return torch.sparse_coo_tensor(coo.coords, coo.data, coo.shape)
   
    
def sparse_expand_as(t1: torch.Tensor, t2:torch.Tensor) -> torch.Tensor:
    return sparse_expand(t1, dims=list(t2.shape))

def sparse_elementwise_div(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    assert (t1.shape == t2.shape), f"Shape of t1 and t2 must match: {t1.shape} != {t2.shape})"
    t1 = t1.coalesce()
    t2 = t2.coalesce()
    t2_inverted = torch.sparse_coo_tensor(t2.indices(), 1/t2.values(), t2.shape)
    return t1 * t2_inverted

def sparse_normalize_old(t: torch.Tensor, dim) -> torch.Tensor:
    # for dim in dims:
    integral = torch.sparse.sum(t, dim=dim).unsqueeze(dim)
    expanded_integral = sparse_expand_as(integral, t)
    result = sparse_elementwise_div(t, expanded_integral)
    return result

def sparse_normalize(x: torch.Tensor, dim) -> torch.Tensor:
    #NOTE: this doesn't work on M1 macos because of numba + sparse
    x = x.cpu().coalesce()
    t = sparse.COO(
        x.indices(), x.values(), shape=x.shape)
    integrals = t.sum(axis=[dim], keepdims=True)
    integrals = sparse.COO(
        integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
    )
    result = t / integrals
    return torch.sparse_coo_tensor(result.coords, result.data, size=result.shape)

if __name__ == "__main__":
    pass

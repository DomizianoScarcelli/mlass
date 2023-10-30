import torch

num_sources = 2
num_tokens = 256
log_prior = torch.randn((num_sources, num_tokens, num_tokens))
log_likelihood = torch.randn((num_tokens, num_tokens, num_tokens))
log_posterior = torch.zeros_like(log_prior)
mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                        254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
mixture_length = mixture.shape[0]

for sample_t in range(mixture_length):
    log_prior_sliced = log_prior[0]
    partial_result = log_prior_sliced + log_likelihood[:, :, sample_t]
    print(f"Partial result shape is {partial_result.shape}")

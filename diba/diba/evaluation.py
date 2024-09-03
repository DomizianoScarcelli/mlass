import torch
from pathlib import Path
import os
import json

def compute_sdr(s_ref: torch.Tensor, s_est: torch.Tensor) -> float:
    # SDR for (s_ref, s_est)
    power_ref = torch.sum(s_ref ** 2)
    power_error = torch.sum((s_ref - s_est) ** 2)
    sdr_1 = 10 * torch.log10(power_ref / power_error)
    
    # SDR for (s_est, s_ref)
    power_est = torch.sum(s_est ** 2)
    power_error_reverse = torch.sum((s_est - s_ref) ** 2)
    sdr_2 = 10 * torch.log10(power_est / power_error_reverse)
    
    # Mean of both SDRs
    sdr_mean = 0.5 * (sdr_1 + sdr_2)
    
    return sdr_mean.item()

def save_sdr(sdr:float, path: Path):
    if not os.path.exists(path):
        content = {"sdr":[]}
    else:
        with open(path, "r") as f:
            content = json.load(f)
    content["sdr"].append(sdr)
    with open(path, "w") as f:
        json.dump(content, f)

def save_psnr(psnr:float, path: Path):
    if not os.path.exists(path):
        content = {"psnr":[]}
    else:
        with open(path, "r") as f:
            content = json.load(f)
    content["psnr"].append(psnr)
    with open(path, "w") as f:
        json.dump(content, f)


def mean_quality(path: Path):
    with open(path, "r") as f:
        content = json.load(f)
    if "psnr" in content:
        values = content["psnr"]
    elif "sdr" in content:
        values = content["sdr"]
    elif "sdr_1" in content and "sdr_2" in content and "sdr_mean" in content:
        values = content["sdr_mean"]
    else:
        raise Exception("No SDR nor PSNR found!")
    mean = torch.mean(torch.tensor(values))
    std = torch.std(torch.tensor(values))
    result = {"mean": mean, "std": std, "num_samples": len(values)}
    save_path = Path(str(path).split(".")[0] + "_mean.json")
    with open(save_path, "w"):
        json.dump(result, f)
    return mean, std


if __name__ == "__main__":
    psnr_gm_2 = Path("/Users/dov/Desktop/wip-projects/latent-autoregressive-source-separation/lass_mnist/psrn_gm_2sources.json")
    psnr_gm_2_raw = Path("/Users/dov/Desktop/wip-projects/latent-autoregressive-source-separation/lass_mnist/psrn_gm_2sources_raw.json")
    mean_quality(psnr_gm_2)
    mean_quality(psnr_gm_2_raw)

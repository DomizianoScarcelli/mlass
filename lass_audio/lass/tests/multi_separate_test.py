import unittest
from pathlib import Path
from lass_audio.lass.datasets import ChunkedMultipleDataset
from lass_audio.lass.multi_separate import SumProductSeparator
import torch
from lass_audio.lass.utils import decode_latent_codes, setup_priors, setup_vqvae, get_raw_to_tokens
from lass_audio.lass.diba_interfaces import JukeboxPrior, SparseLikelihood


class MockData():
    def __init__(self):
        self.NUM_SOURCES = 2
        self.TIME_STEPS = 1024

        audio_root = Path(__file__).parent.parent.parent
        audio_dir_1 = audio_root / "data/bass"
        audio_dir_2 = audio_root / "data/drums"

        SAMPLE_RATE = 44100

        AUDIO_DIRS = [audio_dir_1, audio_dir_2]

        vqvae_path: str = audio_root / "checkpoints/vqvae.pth.tar"
        prior_1_path: str = audio_root / "checkpoints/prior_bass_44100.pth.tar"
        prior_2_path: str = audio_root / "checkpoints/prior_drums_44100.pth.tar"

        sum_frequencies_path: str = audio_root / "checkpoints/sum_frequencies.npz"

        vqvae_type: str = "vqvae",
        prior_1_type: str = "small_prior",
        prior_2_type: str = "small_prior",

        max_sample_tokens: int = 1024,
        sample_rate: int = 44100,

        save_path: str = audio_root / "separated-audio"
        resume: bool = True,
        num_pairs: int = 100,
        seed: int = 0,

        device = torch.device("cpu")

        # setup models
        vqvae = setup_vqvae(
            vqvae_path=vqvae_path,
            vqvae_type=vqvae_type,
            sample_rate=sample_rate,
            sample_tokens=max_sample_tokens,
            device=device,
        )

        priors = setup_priors(
            prior_paths=[prior_1_path, prior_2_path],
            prior_types=[prior_1_type, prior_2_type],
            vqvae=vqvae,
            fp16=True,
            device=device,
        )
        priors = {
            Path(prior_1_path).stem: priors[0],
            Path(prior_2_path).stem: priors[1],
        }

        level = vqvae.levels - 1

        self.separator = SumProductSeparator(
            encode_fn=lambda x: vqvae.encode(
                x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(),  # TODO: check if correct
            decode_fn=lambda x: decode_latent_codes(
                vqvae, x.squeeze(0), level=level),
            priors={k: JukeboxPrior(p.prior, torch.zeros(
                (), dtype=torch.float32, device=device)) for k, p in priors.items()},
            likelihood=SparseLikelihood(sum_frequencies_path, device, 3.0),
        )

        raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)

        self.dataset = ChunkedMultipleDataset(
            instruments_audio_dir=AUDIO_DIRS,
            sample_rate=SAMPLE_RATE,
            max_chunk_size=raw_to_tokens * self.TIME_STEPS,
            min_chunk_size=raw_to_tokens)


class MultiSeparateTest(unittest.TestCase):
    def test_initialize_graphical_model(self):
        data = MockData()
        separator = data.separator

        separator.initialize_graphical_model(
            dataset=data.dataset,
            num_sources=data.NUM_SOURCES)

        # Shape taken from the BeamSearch separator
        # mixture = torch.randn((1, 131072))
        mixture = torch.randn((1, 1024))
        separator.separate(mixture=mixture)


if __name__ == "__main__":
    unittest.main()

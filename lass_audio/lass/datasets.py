import abc
import functools
import warnings
from abc import ABC
from pathlib import Path
from typing import List, Tuple, Union

import librosa
import torch
from torch.utils.data import Dataset

from .utils import get_nonsilent_chunks, load_audio_tracks, load_multiple_audio_tracks, get_multiple_nonsilent_chunks, get_intersection


class SeparationDataset(Dataset, ABC):
    @abc.abstractmethod
    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        ...


class SeparationSubset(SeparationDataset):
    def __init__(self, dataset: SeparationDataset, indices: List[int]):
        self.dataset = dataset
        self.subset = torch.utils.data.Subset(dataset, indices)
        self.indices = indices

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        return self.subset[item]

    def __len__(self) -> int:
        return len(self.subset)

    @property
    def sample_rate(self) -> int:
        return self.dataset.sample_rate


class TrackPairsDataset(SeparationDataset):
    def __init__(
        self,
        instrument_1_audio_dir: Union[str, Path],
        instrument_2_audio_dir: Union[str, Path],
        sample_rate: int,
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = 10.0/221.0

        # Load list of files and starts/durations
        self.dir_1 = Path(instrument_1_audio_dir)
        self.dir_2 = Path(instrument_2_audio_dir)
        dir_1_files = librosa.util.find_files(str(self.dir_1))
        dir_2_files = librosa.util.find_files(str(self.dir_2))

        # get filenames
        dir_1_files = set(sorted([Path(f).name for f in dir_1_files]))
        dir_2_files = set(sorted([Path(f).name for f in dir_2_files]))
        self.filenames = list(sorted(dir_1_files.intersection(dir_2_files)))

        if len(self.filenames) != len(dir_1_files):
            unused_tracks = len(dir_1_files.difference(self.filenames))
            warnings.warn(
                f"Not using all available tracks in {self.dir_1} ({unused_tracks})"
            )

        if len(self.filenames) != len(dir_2_files):
            unused_tracks = len(dir_2_files.difference(self.filenames))
            warnings.warn(
                f"Not using all available tracks in {self.dir_2} ({unused_tracks})"
            )

    def __len__(self):
        return len(self.filenames)

    def get_tracks(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert filename in self.filenames
        track_1, track_2 = load_audio_tracks(
            path_1=self.dir_1 / filename,
            path_2=self.dir_2 / filename,
            sample_rate=self.sr,
        )

        t1_channels, t1_samples = track_1.shape
        t2_channels, t2_samples = track_2.shape
        assert t1_channels == t2_channels == 1
        assert (
            abs(t1_samples - t2_samples) / self.sr <= self.sample_eps
        ), f"{filename}: {abs(t1_samples - t2_samples)}"
        if t1_samples != t2_samples:
            warnings.warn(
                f"The tracks {self.dir_1 / filename} and"
                f"{self.dir_2 / filename} have a different number of samples"
                f"({t1_samples} != {t2_samples})"
            )

        n_samples = min(t1_samples, t2_samples)
        return track_1[:, :n_samples], track_2[:, :n_samples]

    def __getitem__(self, item):
        return self.get_tracks(self.filenames[item])

    @property
    def sample_rate(self) -> int:
        return self.sr


class ChunkedPairsDataset(TrackPairsDataset):
    def __init__(
        self,
        instrument_1_audio_dir: Union[str, Path],
        instrument_2_audio_dir: Union[str, Path],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
    ):
        # Load list of files and starts/durations
        super().__init__(instrument_1_audio_dir, instrument_2_audio_dir, sample_rate)
        self.max_chunk_size = max_chunk_size

        self.available_chunk = {}
        self.index_to_file, self.index_to_chunk = [], []
        for file in self.filenames:
            t1, t2 = self.get_tracks(file)
            available_chunks = get_nonsilent_chunks(
                t1, t2, max_chunk_size, min_chunk_size
            )
            self.available_chunk[file] = available_chunks
            self.index_to_file.extend([file] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_file)

    @functools.lru_cache(1024)
    def load_tracks(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_tracks(filename)

    def __len__(self):
        return len(self.index_to_file)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_file[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        t1, t2 = self.load_tracks(self.get_chunk_track(item))
        t1, t2 = t1[:, chunk_start:chunk_stop], t2[:, chunk_start:chunk_stop]
        return t1, t2


# ---------------------------------------
class TrackMultipleDataset(SeparationDataset):
    def __init__(
        self,
        instruments_audio_dir: List[Union[str, Path]],
        sample_rate: int,
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = 10.0/221.0

        # Load list of files and starts/durations
        self.dirs = [Path(dir_) for dir_ in instruments_audio_dir]
        dir_files = [librosa.util.find_files(str(dir)) for dir in self.dirs]

        dir_filenames = [set(sorted([Path(f).name for f in dir]))
                         for dir in dir_files]

        self.filenames = list(sorted(get_intersection(dir_filenames)))

    def __len__(self):
        return len(self.filenames)

    def get_tracks(self, filename: str) -> List[torch.Tensor]:
        assert filename in self.filenames

        tracks = load_multiple_audio_tracks(
            paths=[dir_ / filename for dir_ in self.dirs],
            sample_rate=self.sr,)

        # t_channels =  [track.shape[0] for track in tracks]
        t_samples = [track.shape[1] for track in tracks]

        # TODO: add back the asserts
        # assert t1_channels == t2_channels == 1
        # assert (
        #     abs(t1_samples - t2_samples) / self.sr <= self.sample_eps
        # ), f"{filename}: {abs(t1_samples - t2_samples)}"
        # if t1_samples != t2_samples:
        #     warnings.warn(
        #         f"The tracks {self.dir_1 / filename} and"
        #         f"{self.dir_2 / filename} have a different number of samples"
        #         f"({t1_samples} != {t2_samples})"
        #     )

        n_samples = min(t_samples)
        return [track[:, :n_samples] for track in tracks]

    def __getitem__(self, item) -> List[torch.Tensor]:
        return self.get_tracks(self.filenames[item])

    @property
    def sample_rate(self) -> int:
        return self.sr


class ChunkedMultipleDataset(TrackMultipleDataset):
    def __init__(
        self,
        instruments_audio_dir: List[Path],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
    ):
        # Load list of files and starts/durations
        super().__init__(instruments_audio_dir, sample_rate)
        self.max_chunk_size = max_chunk_size

        self.available_chunk = {}
        self.index_to_file, self.index_to_chunk = [], []
        for file in self.filenames:
            tracks = self.get_tracks(file)
            available_chunks = get_multiple_nonsilent_chunks(
                tracks, max_chunk_size, min_chunk_size
            )
            self.available_chunk[file] = available_chunks
            self.index_to_file.extend([file] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_file)

    @functools.lru_cache(1024)
    def load_tracks(self, filename: str) -> List[torch.Tensor]:
        return self.get_tracks(filename)

    def __len__(self):
        return len(self.index_to_file)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_file[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> List[torch.Tensor]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        tracks = self.load_tracks(self.get_chunk_track(item))
        tracks = [t[:, chunk_start:chunk_stop] for t in tracks]
        return tracks

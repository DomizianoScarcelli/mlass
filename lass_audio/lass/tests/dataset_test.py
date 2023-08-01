from ..datasets import TrackMultipleDataset, TrackPairsDataset, ChunkedMultipleDataset, ChunkedPairsDataset
from pathlib import Path
import torch

audio_root = Path(__file__).parent.parent.parent

audio_dir_1 = audio_root / "data/bass"
audio_dir_2 = audio_root / "data/drums"

AUDIO_DIRS = [audio_dir_1, audio_dir_2]
SAMPLE_RATE = 44100


def check_datasets_equality(dataset1, dataset2):
    # Step 1: Ensure the datasets have the same length
    if len(dataset1) != len(dataset2):
        print(
            f"Datasets have differnt lengths. dataset1: {len(dataset1)} dataset2: {len(dataset2)}")
        return False

    # Step 2: Iterate through the datasets and compare the items
    for i in range(len(dataset1)):
        item1 = tuple(item.tolist() for item in dataset1[i])
        item2 = tuple(item.tolist() for item in dataset2[i])

        # You might need to customize this comparison based on your data type
        if item1 != item2:
            print(
                f"""Datasets differ on at least one item.
            Item1: {item1}
            Item2: {item2}
            """)
            return False

    return True


def test_pair_datasets():
    pair_dataset = TrackPairsDataset(
        instrument_1_audio_dir=audio_dir_1,
        instrument_2_audio_dir=audio_dir_2,
        sample_rate=SAMPLE_RATE)
    multiple_dataset = TrackMultipleDataset(
        instruments_audio_dir=AUDIO_DIRS,
        sample_rate=SAMPLE_RATE)

    assert (check_datasets_equality(pair_dataset, multiple_dataset)), f"""
    Datasets are different!
    pair_dataset: {pair_dataset}
    multiple_dataset: {multiple_dataset}
    """


def test_get_multiple_non_silent_chunks():
    # This is implicit since if test_chunk_datasets works, then this has to work too.
    raise NotImplementedError()


def test_chunk_datasets():
    chunk_pair_dataset = ChunkedPairsDataset(
        instrument_1_audio_dir=audio_dir_1,
        instrument_2_audio_dir=audio_dir_2,
        sample_rate=SAMPLE_RATE,
        max_chunk_size=100,
        min_chunk_size=10)
    chunk_multiple_dataset = ChunkedMultipleDataset(
        instruments_audio_dir=AUDIO_DIRS,
        sample_rate=SAMPLE_RATE,
        max_chunk_size=100,
        min_chunk_size=10)

    assert (check_datasets_equality(chunk_pair_dataset, chunk_multiple_dataset)), f"""
    Datasets are different!
    pair_dataset: {chunk_pair_dataset}
    multiple_dataset: {chunk_multiple_dataset}
    """


if __name__ == "__main__":
    test_pair_datasets()
    test_chunk_datasets()

"""
TODO: 
I'm dumb and I don't know how python imports for local packages work, so in order to execute this
place yourself in the root and run `python -m lass_audio.lass.tests.utils_test`. In the future if I understand how modules
work this will be runnable using `pytest`.
"""

from ..utils import load_audio_tracks, load_multiple_audio_tracks, get_nonsilent_chunks, get_multiple_nonsilent_chunks
from pathlib import Path

audio_root = Path(__file__).parent.parent.parent
path_1 = audio_root / "data/bass/Track01876.wav"
path_2 = audio_root / "data/drums/Track01876.wav"

PATHS = [path_1, path_2]
SAMPLE_RATE = 44100


def test_multiple_audio_tracks():
    """
    Tests if, for 2 tracks, load_audio_tracks and load_multiple_audio_tracks produce the same output.
    """
    path_1, path_2 = PATHS[0], PATHS[1]
    audio_tracks = load_audio_tracks(
        path_1=path_1, path_2=path_2, sample_rate=SAMPLE_RATE)
    multiple_audio_tracks = load_multiple_audio_tracks(
        paths=PATHS, sample_rate=SAMPLE_RATE)

    assert audio_tracks[0].tolist() == multiple_audio_tracks[0].tolist() and audio_tracks[1].tolist() == multiple_audio_tracks[1].tolist(), f"""
        test_multiple_audio_tracks failed:
        audio_tracks: {audio_tracks} with type {type(audio_tracks)} 
        multiple_audio_tracks: {multiple_audio_tracks}w ith type {type(multiple_audio_tracks)} 
        """


def test_get_multiple_nonsilent_chunks():

    track_1, track_2 = load_audio_tracks(
        path_1=path_1, path_2=path_2, sample_rate=SAMPLE_RATE)

    chunks = get_nonsilent_chunks(track_1=track_1,
                                  track_2=track_2)

    raise NotImplementedError()


if __name__ == "__main__":
    test_multiple_audio_tracks()
    test_get_multiple_nonsilent_chunks()

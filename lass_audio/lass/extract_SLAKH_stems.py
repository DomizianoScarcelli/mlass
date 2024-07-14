from os.path import exists
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import shutil

audio_root = Path(__file__).parent.parent
data_dir = audio_root / "data" 
slakh_path = data_dir / "SLAKH"
baby_slakh_path = data_dir / "babyslakh_16k"

def extract_stem(path: Path, desired_stem: str):
    new_stem_dir = data_dir / "extracted_stems"/ desired_stem
    os.makedirs(new_stem_dir, exist_ok=True)
    for track in tqdm(os.listdir(path), f"Extracting stem: {desired_stem}"):
        full_path = path / track
        if not os.path.isdir(full_path):
            continue
        with open(full_path / "metadata.yaml", "r") as f:
            metadata = yaml.safe_load(f)
        print("Metadata: ", metadata)
        for stem in metadata["stems"]:
            stem_metadata = metadata["stems"][stem]
            stem_name = stem_metadata["inst_class"].lower().replace("(continued)", "").strip()
            print("Stem name: ", stem_name)
            if stem_name != desired_stem:
                continue
            src = str(full_path / "stems" / stem) + ".wav"
            dst = str(new_stem_dir / track) + ".wav"
            try:
                shutil.copyfile(src, dst)
            except FileNotFoundError as e:
                print(f"File {src} not found, skipping...")


def run():
    extract_stem(path=baby_slakh_path, desired_stem="piano")

if __name__ == "__main__":
    run()



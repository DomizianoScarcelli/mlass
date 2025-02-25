from os.path import exists
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import shutil

audio_root = Path(__file__).parent.parent
data_dir = audio_root / "data" 
external_hhd_dir = Path("/Volumes/Seagate HDD/Brave/slakh2100_flac_redux/train")
external_hdd_output_dir = Path("/Volumes/Seagate HDD/Brave/slakh processed/train")
slakh_path = data_dir / "SLAKH"
baby_slakh_path = data_dir / "babyslakh_16k"
FORMAT = ".flac" 

def extract_stem(path: Path, desired_stem: str):
    # new_stem_dir = data_dir / "extracted_stems"/ desired_stem
    new_stem_dir = external_hdd_output_dir / "flac_stems" / desired_stem
    os.makedirs(new_stem_dir, exist_ok=True)
    for track in tqdm(os.listdir(path), f"Extracting stem: {desired_stem}"):
        full_path = path / track
        if not os.path.isdir(full_path):
            continue
        with open(full_path / "metadata.yaml", "r") as f:
            metadata = yaml.safe_load(f)
        # print("Metadata: ", metadata)
        for stem in metadata["stems"]:
            stem_metadata = metadata["stems"][stem]
            stem_name = stem_metadata["inst_class"].lower().replace("(continued)", "").strip()
            # print("Stem name: ", stem_name)
            if stem_name != desired_stem:
                continue
            src = str(full_path / "stems" / stem) + FORMAT
            dst = str(new_stem_dir / track) + FORMAT
            try:
                shutil.copyfile(src, dst)
            except FileNotFoundError as e:
                print(f"File {src} not found, skipping...")


def run():
    extract_stem(path=external_hhd_dir, desired_stem="piano")
    extract_stem(path=external_hhd_dir, desired_stem="drums")
    extract_stem(path=external_hhd_dir, desired_stem="bass")

if __name__ == "__main__":
    run()



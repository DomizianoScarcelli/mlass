# LASS
This is a custom README file for the paper *Latent Autoregressive Source Separation*, for the
original readme of this repo see [here](README_old.md)

## Installation
Before running this code it is necessary to install an MPI implementation. On ubuntu, this can be done
simply by running the command
```bash
sudo apt-get -y install mpich
``` 
Installing the necessary dependencies can be done through conda, by executing the following commands:
```bash
cd lass_audio
conda env create -f environment.yaml
conda activate lass_audio
``` 
Once the conda environment is installed it is possible to start the separation procedure.

## Download Pre-trained models and data
You can download the necessary checkpoints from [here](https://drive.google.com/file/d/1R50broLWAZwy0qSVezQRTy7EErQPHqnG/view?usp=share_link).
Place the downloaded file inside of the directory `lass_audio/checkpoints` and extract it with the command `tar -xf lass-slakh-ckpts.tar`.

The Slakh2100 test data can be found [here](https://drive.google.com/file/d/1Gf5SHVb8_o5NMbJAWaoULianX8RxzL4L/view?usp=share_link).
Similarly to before, place the downloaded file inside of the directory `lass_audio/data` and extract it with the command `tar -xf bass-drums-slakh.tar`

## Separate images
### Belief propagation via graphical model
To run the separation using belief propagation via a graphical model, put yourself in the root of the repository and run the script

```sh
python -m lass_audio.lass.graphical_model_separate
```

For 2 sources separation

```sh
python -m lass_audio.lass.graphical_model_separate_three_sources
```

For 3 sources separation

This will separate the test set of mnist into the folder `./results/separation/graphical-model/2-sources` for two sources and `./results/separation/graphical-model/3-sources` for three sources.

The SDR for will be logged in the `./results/metrics/` folder

---

Note that each time a separation method is called, the corrisponding directory is emptied.

# LASS

## Installation
Installing the necessary dependencies can be done through conda, by executing the following commands:
```bash
cd lass_mnist
conda env create -f environment.yaml
conda activate lass_mnist
``` 
Once the conda environment is installed it is possible to start the separation procedure.

## Download Pre-trained models
You can download the necessary checkpoints from [here](https://drive.google.com/file/d/1oayY1FEUrTwQJMr78mP1t6r8AggjzAso/view?usp=share_link). 
Place the downloaded file inside of the directory `lass_mnist/checkpoints` and extract it with the command `tar -xf lass-mnist-ckpts.tar`.

## Separate images
### Belief propagation via graphical model
To run the separation using belief propagation via a graphical model, put yourself in the root of the repository and run the script

```sh
python -m lass_mnist.lass.graphical_model_separate
```

For 2 sources separation

```sh
python -m lass_mnist.lass.graphical_model_separate_three_sources
```

For 3 sources separation

This will separate the test set of mnist into the folder `separated-images/graphical-model/2-sources` for two sources and `separated-images/graphical-model/3-sources` for three sources.

### Probabilistic extractor

To run the separation using the probabilistic extractor, put yourself in the root of the repository and run the script

```sh
python -m lass_mnist.lass.pe_separate
```

For 2 sources separation

```sh
python -m lass_mnist.lass.pe_separate_three_sources
```

For 3 sources separation

This will separate the test set of mnist into the folder `separated-images/probabilistic-extractor/2-sources` for two sources and `separated-images/probabilistic-extractor/3-sources` for three sources.

---

Note that each time a separation method is called, the corrisponding directory is emptied.

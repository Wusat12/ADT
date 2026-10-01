# Automatic Drum Transcription: Generalization via Architecture and Dataset

This repository contains all the source code for my Master's thesis: "Automatic Drum Transcription: Generalization via Architecture and Dataset". 

## Virtual environment

Ensure you have [Python 3.11](https://www.python.org/) installed.

It is highly recommended to create a Python virtual environment:

```sh
python -m venv .venv
```

Activate the environment, and download all external libraries:

```sh
pip install -r requirements.txt
```

## Project structure

* ```run.py``` - Train specific models on specific datasets
* ```test.py``` - Test already trained models on all datasets
* ```study/``` - Contains information of all models trained and selected for the thesis, void of specific model weights
* ```thesis/``` - Contains the LaTeX for the thesis
* ```models/``` - Contains all models trained in the thesis
* ```io/``` - Contains scripts used to parse all dataset into PyTorch
* ```notebooks/``` - Contains notebooks used for analysis, or creating figures for the thesis
* ```train.py``` - Contains the training function used for training models
* ```evaluate.py``` - Contains functions used to evaluate the models
* ```preprocess.py``` - Contains functions used for data preprocessing

## Datasets

Datasets need to be explicitly stored under a ```data``` directory in the root folder.

Inside this directory, each dataset has to be stored as follows:
```
data/
├─ adtof/                   # containing each downloaded dataset directory
├─ ENST+MDB/                # containing each downloaded dataset directory
├─ e-gmd-v1.0.0/            # as downloaded, containing .csv file
├─ slakh2100_flac_redux/    # as downloaded
└─ SADTP/                   # as downloaded
```

Each dataset can be found here:
* [ENST-Drums](https://zenodo.org/records/7432188)
* [MDB Drums](https://github.com/CarlSouthall/MDBDrums)
* [E-GMD](https://zenodo.org/records/4300943)
* [Slakh](https://zenodo.org/records/4599666)
* [SADTP](https://github.com/RunarFosse/SADTP)

Each dataset can then be successfully loaded by running each corresponding conversion file, for example:
```sh
python io/convert_sadtp.py
```

## RayTune specific

Due to training with RayTune, if running over several GPUs, limit the scope to one GPU using the environment variable:
```sh
export CUDA_VISIBLE_DEVICES=0
```

This restricts RayTune's access to only this GPU (here the first GPU, GPU0).

## Train a model

To see all parameters for ```run.py```, run:
```sh
python run.py --help
```

An example is:
```sh
CUDA_VISIBLE_DEVICES=5 python run.py --model=crnn --dataset=egmd
```

One can also train a model on several datasets:
```sh
CUDA_VISIBLE_DEVICES=5 python run.py --model=vit --dataset egmd slakh adtof_yt
```

To run on the CPU, explicitly run with CPU set as in this example:
```sh
python run.py cpu --model=cnn --dataset=enst+mdb
```

Afterwards, an output:
* ```config.pt``` file, containing the selected, best-performing model's configs
* ```metrics.csv``` file, containing training metrics such as loss values
* ```model.pt``` file, containing model weights

These will appear under the respective path ```study/Architecture/<model>/<dataset>```, if trained on one dataset, or ```study/Dataset/<model>/<datasets>```, if trained on multiple.

## Test a model

To see all parameters for ```test.py```, run:
```sh
python test.py --help
```

To test a model, specify an already trained model's directory:
```sh
python test.py --path=study/Architecture/Convolutional\ RNN/Slakh/
```

Afterwards, an output ```tests.txt``` file will appear in the model's directory, with the resulting performances.
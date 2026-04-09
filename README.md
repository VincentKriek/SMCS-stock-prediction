# SMCS-stock-prediction

This is the code for the research project **Predicting Stock Prices Using News Articles and
a Data-Driven GNN**. This project is run using UV. For more information about the UV package manager follow [this link](https://docs.astral.sh/uv/). The project can be initiated by running the following two commands from the project root:

```sh
$ uv venv
$ uv run sync.py
```

The data used for this code is retrieved from multiple sources. The readme's in every package will explain which data is used for that package and how the code can be run. The data used in this research is the following:

* [FNSPID dataset](https://huggingface.co/datasets/Zihan1004/FNSPID)
* [Datasets created for/by this project, based on the FNSPID dataset](https://huggingface.co/datasets/VincentKriek/LLM-S-P-500-Dataset)
* [SEC 13F filing data](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets)
* [SEC fails to deliver data](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data)
* [CUSIP.csv mapping to stock symbol from Stock_Data repo](https://github.com/yoshishima/Stock_Data)

The sync file and different packages are explained below. All code can be run to validate our work, but for each package the files necessary to begin are mentioned at the top of the readme. All these files can be found in the sources mentioned above.

## environment variables

Create a `.env` file in the project root. This file should contain at least the following variables:

```env
HF_REPO_ID="Zihan1004/FNSPID" # id of the huggingface repo for loading FNSPID data
HF_SUBFOLDER="Stock_price" # subfolder from which the data should be loaded. Supported values are either Stock_price or Stock_news
MIN_DATE="2018-01-01" # minimum date that will be loaded
MAX_DATE="2023-12-31" # maximum date that will be loaded
```

## sync.py

The sync.py will install all the right packages, including the right pytorch packages for the installed hardware. Meaning, if you have a gpu installed, this file will detect it and install the right packages accordingly.

## data_loader

The data_loader is the first step of the process. It will prepare the datasets for the data_preprocessing step. More information can be found [here](src/data_loader/readme.md).

## data_preprocessing

The data_preprocessing step fully prepares the base dataset for the model run. Here the LLM sentiment scores are generated and potential gaps in the sentiment score are filled in. More information can be found [here](src/data_preprocessing/readme.md)

## model

The model is the final model with graph creation for the MDGNN model, running the actual experiment setup, the baselines and the analysis. More information can be found [here](src/model/readme.md)

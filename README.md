# MolBART - aka Chemformer

The MolBART project aims to pre-train a BART transformer language model [[1]](#1) on molecular SMILES strings [[2]](#2) by optimising a de-noising objective. We hypothesised that pre-training will lead to improved generalisation, performance, training speed and validity on downstream fine-tuned tasks. We tested the pre-trained model on downstream tasks such as reaction prediction, retrosynthetic prediction, molecular optimisation and molecular property prediction.

We have now published our results in a pre-print [[3]](#3) which was accepted in Machine Learning: Science and Technology [[4]](#4) and will make the models and datasets available [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq).


## Installation

The project dependencies can be installed as follows:
- `conda create --name molbart rdkit -c rdkit`
- `conda activate molbart`
- `conda install pytorch==1.8.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia`
- `conda install gcc_linux-64 gxx_linux-64 mpi4py`
- `pip install -r requirements.txt`

[pysmilesutils](https://github.com/MolecularAI/pysmilesutils) must also be downloaded and installed. It can be done directly with pip
- `python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git`


## Running the Code

The following is an example run through of how to run the Chemformer code on the pre-trained models and datasets available [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq).

1. Create a Chemformer conda environment, as above.
1. Download the dataset of interest and store it locally (let's say ../data/uspto_50.pickle).
1. Download a pre-trained Chemformer model and store it locally (let's say ../models/pre-trained/combined.ckpt).
1. Update the `fine_tune.sh` shell script in the example_scripts directory (or create your own) with the paths to your model and dataset, as well as the values of hyperparameters you wish to pass to the script.
1. Run the `fine_tune.sh` script.

You can of course run other scripts in the repository following a similar approach. The Scripts section below provides more details on what each script does.


## Codebase

The codebase is broadly split into the following parts:
* Models
* Data helpers
* Tokenisation
* Decoding
* Scripts

### Models

The  `models.py` file contains a Pytorch Lightning implementation of the BART language model, as well as Pytorch Lightning implementations of models for downstream tasks.

### Data Helpers

The `dataset.py` file contains a number of classes used to load, batch and process the data before it is passed to the model. Classes which inherit from `_AbsDataset` are subclasses of Pytorch's `nn.utils.Dataset` and are simply used to store and split data (molecules, reactions, etc) into its relevant subset (train, val, test).

Our `_AbsDataModule` class inherits from Pytorch Lightning's `LightningDataModule` class, and its subclasses are used to augment, tokenise and tensorise the data before it passed to the model.

Finally, we include a `TokenSampler` class which categorises sequences into buckets based on their length, and is able to sample a different batch size of sequences from each bucket. This helps to ensure that the model sees approximately the same number of tokens on each batch, as well as dramatically improving training speed.

### Tokenisation

Our `tokenise.py` file includes the `MolEncTokeniser` class which is capable of random 'BERT-style' masking of tokens, as well as padding each batch of sequences to be the same length. The tokeniser makes use of the `SMILESTokenizer` from the `pysmilesutils` library for tokenising SMILES into their constituent atoms.

### Decoding

We include implementations of greedy and beam search decoding in the `decode.py` file. Both implementations make use of batch decoding for improved evaluation speeds. They do not, however, cache results from previous decodes, rather, they simply pass the entire sequence of tokens produced so far through the transformer decoder.

### Scripts

The repository includes the following scripts:
* `train.py` runs the pre-training 
* `fine_tune.py` runs fine-tuning on a specified task
* `evaluate.py` evaluates the performance of a fine-tuned model
* `predict.py` writes the SMILES outputs of the model to a pandas DF stored in a pickle file
* `build_tokeniser.py` creates a tokeniser from a dataset and stores it in a pickle file

Each script can be run using `python -m molbart.<scipt_name> <args>`.

See the ArgumentParser args in each file for more details on each argument.

To run on multiple GPUs use the `--gpus <num>` argument for the train or fine tune scripts. This will run the script with Pytorch Lightning's distributed data parallel (DDP) processing. Validation will be disabled when using DDP to ensure the GPUs stay synchronised and stop possible deadlocks from occurring.


## References

<a id="1">[1]</a>
Lewis, Mike, et al.
"Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension."
arXiv preprint arXiv:1910.13461 (2019).

<a id="2">[2]</a>
Weininger, David.
"SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules."
Journal of chemical information and computer sciences 28.1 (1988): 31-36.

<a id="3">[3]</a>
Irwin, Ross, et al.
"Chemformer: A Pre-Trained Transformer for Computational Chemistry."
ChemRxiv (2021). doi:10.33774/chemrxiv-2021-v2pnn

<a id="4">[4]</a>
Irwin, R., Dimitriadis, S., He, J., Bjerrum, E.J., 2021. Chemformer: A Pre-Trained Transformer for Computational Chemistry. Mach. Learn. Sci. Technol. [https://doi.org/10.1088/2632-2153/ac3ffb](https://doi.org/10.1088/2632-2153/ac3ffb)

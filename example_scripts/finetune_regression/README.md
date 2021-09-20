# Training of regression models from pre-trained MolBART models



The `finetune_regression_modules.py` file contains classes of the Data Helpers and Models for training the Chemformer on downstream Regression tasks. Most of these classes inherit from classes which are created in `molbart` implementation.

The scripts `trainRegr.py`, `finetuneRegr.py` are for training and fine-tuning respectively; the Encoder part of the Chemformer on a number of regression tasks simultaneously.

Each script can be run using `python -m molbart.example_scripts.finetune_regression.<script_name> <args>`.

See the ArgumentParser args in each file for more details on each argument.
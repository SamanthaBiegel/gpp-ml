
This repository contains the code for experiments and figures for the paper "Unrecognised water limitation is a main source of uncertainty for
models of terrestrial photosynthesis".

To reproduce the experiments, follow these steps:

# Install environment with conda
```
conda create -n gpp-ml python=3.10.12
conda activate gpp-ml
pip install -r requirements.txt
```

**Note:** Install torch with cuda if applicable on your system.

# Run model experiments
```
python -u global_model.py --model_type LSTM --early_stopping --layernorm
python -u global_model.py --model_type MLP --early_stopping
python -u site_specific_model --model_type LSTM --early_stopping --layernorm
python -u site_specific_model --model_type MLP --early_stopping
python -u global_model.py --model_type LSTM --early_stopping --layernorm --extra_features
python -u global_model.py --model_type MLP --early_stopping --extra_features
```

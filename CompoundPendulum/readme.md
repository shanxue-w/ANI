# Damped Compound Pendulum

Note for the Damped Compound Pendulum experiment: `dataset/generate_compound_dataset.m`
is the dataset-generation script for this experiment. The saved train/validation
MAT files contain 1,000 one-step training pairs and 1,000 one-step validation
pairs, matching the split described in the Supplementary Information.

`dataset/data.py` converts these MAT files to the processed NumPy files used by
the training scripts. The Python training loaders also keep `[:1000, :]`, so the
reported training and validation sizes remain 1,000 pairs each.

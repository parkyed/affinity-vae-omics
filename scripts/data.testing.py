# testing the data loader function in isolation

import numpy as np
from avae.data import load_data

# ========== testing running the data loader ===============

np.random.seed(42)

datapath = "/Users/ep/Documents/1_datasets/aff_vae/affinity-vae-omics/omics/omics_data/input_arrays/"
affinitypath = "/Users/ep/Documents/1_datasets/aff_vae/affinity-vae-omics/omics/omics_data/affinity_omics.csv"

trains, vals, tests, lookup, data_dim = load_data(
    datapath=datapath,
    datatype='npy',
    # lim=lim,
    splt= 20,
    batch_s= 10,
    eval=False,
    affinity=affinitypath,
    classes=None,
    # normalise=True
)

next(iter(trains))
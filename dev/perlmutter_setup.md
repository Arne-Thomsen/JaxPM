Setup in `.bash_profile`:
```
module load cpe/24.07
module load python/3.11
module load cudatoolkit/12.9
conda activate flatiron
```

Setup of the conda environment:
```
module load python
conda create -n flatiron python=3.11 pip numpy scipy rich
conda activate flatiron
pip install --upgrade jax[cuda12]
pip install ipykernel tqdm numpy scipy matplotlib diffrax flax optax dm-haiku orbax h5py jraph hdf5plugin seaborn
python -m ipykernel install --user --name flatiron --display-name flatiron
~/flatiron/JaxPM> pip install -e .
```
```
module load python_cuda/3.11.6
python -m venv --system-site-packages flatiron
source flatiron/bin/activate
python -m pip install CAMELS-library dm-haiku diffrax optax
python -m pip install --no-deps flax orbax-checkpoint
python -m pip install tensorstore importlib_resources
python -m pip install tensorflow-probability[jax]==0.24.0

python -m pip install ipykernel
python -m ipykernel install --user --name flatiron --display-name flatiron
```
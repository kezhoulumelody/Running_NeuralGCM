# Running NeuralGCM on delphi server 

## Setting up the environment 
We first set up a conda environment and install necessary dependencies. 

```bash
cd /data/<user>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/<user>/miniconda3

# Initialize conda from custom location
source /data/<user>/miniconda3/etc/profile.d/conda.sh

source ~/.bashrc

# Create your ML environment inside /data
conda create -p /data/<user>/mlclimate python=3.11 -y
conda activate /data/<user>/mlclimate

# Install basic libraries
conda install -y \
    numpy scipy pandas xarray netcdf4 \
    matplotlib jupyterlab pip

# CUDA 13 GPU JAX
pip install --upgrade "jax[cuda13]"
```
> [!NOTE]
> We need to check if JAX is correctly installed by running the following python command:

```python
import jax, jax.numpy as jnp
print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

x = jnp.arange(5.0)
print("x^2:", x**2)

for d in jax.devices():
    print("Device:", d, "platform:", d.platform)
```
Correctly installed JAX should give output similar to:

Device: cuda:0 platform: gpu

Device: cuda:1 platform: gpu

## Add config files into .bashrc

```bash
mkdir -p /data/<user>/.cache/jax
mkdir -p /data/<user>/.cache
mkdir -p /data/<user>/.config/matplotlib

echo 'export XDG_CACHE_HOME=/data/<user>/.cache' >> ~/.bashrc
echo 'export MPLCONFIGDIR=/data/<user>/.config/matplotlib' >> ~/.bashrc
echo 'export JAX_CACHE_DIR=/data/<user>/.cache/jax' >> ~/.bashrc

# Now reload the shell
source ~/.bashrc
```
> [!NOTE]
> If you are using VSCode, the ipykernel also needs to be installed

```bash
pip install ipykernel
python -m ipykernel install --user --name /data/<user>/mlclimate --display-name "Python (mlclimate)"
```
In VSCode, Cmd + Shift + P to open Command Palette, and search for "Python: Select Interpreter" and then choose corresponding interpreter.

> [!WARNING]
> Since conda is installed in /data/ instead of default home directory, extra steps are needed to load conda evertime you login.

```bash
echo 'source /data/kezhoulumelody/miniconda3/etc/profile.d/conda.sh' >> ~/.bash_profile
source ~/.bash_profile
```

## Install NeuralGCM
Next we install the Neural GCM from https://neuralgcm.readthedocs.io/en/stable/installation.html

```bash
pip install neuralgcm
```

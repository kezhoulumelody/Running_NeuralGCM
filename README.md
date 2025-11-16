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
pip install --upgrade "jax[cuda13] flax dm-haiku optax"
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

## Running NeuralGCM

First load necessary libs:

```python
import gcsfs
import jax
import numpy as np
import pickle
import xarray

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm
```

Load a pre-trained model:

```python
model_name = 'v1/deterministic_0_7_deg.pkl'  #@param ['v1/deterministic_0_7_deg.pkl', 'v1/deterministic_1_4_deg.pkl', 'v1/deterministic_2_8_deg.pkl', 'v1/stochastic_1_4_deg.pkl', 'v1_precip/stochastic_precip_2_8_deg.pkl', 'v1_precip/stochastic_evap_2_8_deg.pkl'] {type: "string"}

gcs = gcsfs.GCSFileSystem(token='anon')
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
```

Load ERA5 data from GCP/Zarr and regrid it to model's native grids:

```python
vera5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(
    era5_path, chunks=None, storage_options=dict(token='anon')
)

demo_start_time = '2009-02-14'
demo_end_time = '2009-02-18'
data_inner_steps = 24  # process every 24th hour

sliced_era5 = (
    full_era5
    [model.input_variables + model.forcing_variables]
    .pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift='24 hours',
    )
    .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
    .compute()
)

## Regrid to NeuralGCM’s native resolution:
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
```
Prepare all-forcing files. Here we use a constant foring file as example. The forcing file contains SST and sea ice concentration

```python
## Prepare climatological forcing data: Here we use 1979 to 2019

era_1979_2019 = full_era5.sel(time=slice("1979-01-01", "1979-01-31"))
full_forcing_select = era_1979_2019[["sea_ice_cover", "sea_surface_temperature"]]
forcing_data_slice = full_forcing_select.mean(dim="time", keep_attrs=True) 

# Add back single time coordinate
forcing_clim = forcing_data_slice.expand_dims(time=[full_forcing_select.time.values[0]])

# Regrid forcing file
forcing_ngcm28 =  xarray_utils.regrid(forcing_clim, regridder)
forcing_ngcm28 = xarray_utils.fill_nan_with_nearest(forcing_ngcm28)
```
Run the model, save daily output and restart files for every 30 days:

```python

# ==============================
#   CONFIG
# ==============================

RESTART_ROOT = "/data/xxx/xxx/NeuralGCM_restart/ngcm28_restarts_test"
restart_step = 30  # the folder we saved previously

nyears = 1
ndays = 30
inner_steps = 24  # model internal step = 24 hours
outer_steps = nyears * ndays * 24 // inner_steps  # = 30 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # hours since t0

os.makedirs(RESTART_ROOT, exist_ok=True)

# ==============================
#   INITIAL STATE (from ERA5)
# ==============================

inputs = model.inputs_from_xarray(eval_ngcm28.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_ngcm28.isel(time=0))
rng_key = jax.random.key(42)  # important for stochastic model
initial_state = model.encode(inputs, input_forcings, rng_key)

# ==============================
#   FORCINGS FOR THIS SEGMENT
# ==============================
all_forcings = model.forcings_from_xarray(forcing_ngcm28)


# ==============================
#   RUN MODEL
# ==============================
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
) 

predictions_ds = model.data_to_xarray(predictions, times=times)
predictions_ds.to_netcdf("/data/xxx/xxx/NeuralGCM_output/Testing/ngcm28_deterministic-run_segment_0001.nc")

# ==============================
#   SAVE RESTART FILES
# ==============================

# we'll label this restart by the step index we reached (30)
step_index = outer_steps 

ckpt_path = os.path.join(RESTART_ROOT, str(step_index))  # ".../30"

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_path, final_state)

print(f"Saved restart at step {step_index} in {ckpt_path}")

```

## Restart a run

There are two ways to restart a run, one from the checkpointer files and the other directly from final state.

The easy way is to start from final_state (assuming final_state is correctly encoded)

```python
# Time setup for the next 30 days
inner_steps = 24                     # same as before
extra_ndays = 30
extra_outer_steps = extra_ndays * 24 // inner_steps   # = 30

timedelta = np.timedelta64(1, 'h') * inner_steps
start_step = restart_step          # 30
end_step = restart_step + extra_outer_steps  # 60

# Time coordinate for the *new* segment (still in hours since t0)
times_segment = np.arange(start_step, end_step) * inner_steps
# state_new = model.encode(final_state, input_forcings, rng_key)

# Continue the forecast from the restored state
final_state_3, predictions_3 = model.unroll(
    final_state,
    all_forcings,
    steps=extra_outer_steps,
    timedelta=timedelta,
    start_with_input=True,   # IMPORTANT: do not set it to FALSE to prevent timestamps mismatch
)

# Convert to xarray
predictions_3_ds = model.data_to_xarray(predictions_3, times=times_segment)
predictions_3_ds.to_netcdf("/data/xxx/xxx/NeuralGCM_output/Testing/ngcm28_deterministic-run_segment_0002-directly-from-final-state.nc")
```

We can also start from saved restart file. Note, restart file is NOT directly saved as netCDF for xarry data format.

```python
# ==============================
#   LOAD RESTART FILES
# ==============================

## Restore the state at step 30 
RESTART_ROOT = "/data/kezhoulumelody/melody_data/NeuralGCM_restart/ngcm28_restarts_test"

# We stopped the first run at step 30
restart_step = 30

ckpt_path = os.path.join(RESTART_ROOT, str(restart_step))

# # ---------- build abstract (template) state ----------
inputs0   = model.inputs_from_xarray(eval_ngcm28.isel(time=0))
forcings0 = model.forcings_from_xarray(eval_ngcm28.isel(time=0))
rng0      = jax.random.key(42)

dummy_state = model.encode(inputs0, forcings0, rng0)

abstract_state = jax.tree_util.tree_map(
    ocp.utils.to_shape_dtype_struct, dummy_state
)

# ---------- restore ----------
checkpointer = ocp.StandardCheckpointer()
state = checkpointer.restore(ckpt_path, abstract_state)

# ==============================
#   CONTINUE THE RUN
# ==============================

# Time setup for the next 30 days
inner_steps = 24                     # same as before
extra_ndays = 30
extra_outer_steps = extra_ndays * 24 // inner_steps   # = 30

timedelta = np.timedelta64(1, 'h') * inner_steps
start_step = restart_step           # 30
end_step = restart_step + extra_outer_steps  # 60

# Time coordinate for the *new* segment (still in hours since t0)
times_segment = np.arange(start_step, end_step) * inner_steps

# Continue the forecast from the restored state
final_state_2, predictions_2 = model.unroll(
    state,
    all_forcings,
    steps=extra_outer_steps,
    timedelta=timedelta,
    start_with_input=True,   # IMPORTANT: we're continuing, not restarting from ERA5
)

# Convert to xarray
predictions_2_ds = model.data_to_xarray(predictions_2, times=times_segment)
predictions_2_ds.to_netcdf("/data/xxx/xxx/NeuralGCM_output/Testing/ngcm28_deterministic-run_segment_0002-part2.nc")
```


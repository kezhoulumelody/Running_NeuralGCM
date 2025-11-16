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

# create your ML environment inside /data




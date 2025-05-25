# Pipeline for setting up Mistral / Pixtral-12b on BW Uni Cluster 3.0 (Example setup)

This repository demonstrates how to set up an account, prepare an environment, run inference on simple “dot” spatial-relation tasks and evaluate results on the BW Uni Cluster 3.0 with a GPU.

## 📋 Table of Contents

1. [Account Setup](#account-setup)  
2. [Cluster Environment Preparation](#cluster-environment-preparation)  
3. [Data & Directory Layout](#data--directory-layout)  
4. [Inference Script & Slurm Submission](#inference-script--slurm-submission)  
5. [Evaluation Script](#evaluation-script)  
6. [Troubleshooting & Tips](#troubleshooting--tips)  


## 1. Account Setup and Login

1. Follow these steps to create an account at BW Uni Cluster 3.0:  https://wiki.bwhpc.de/e/Registration/bwUniCluster
2. Download and install MobaXterm here: https://mobaxterm.mobatek.net/
3. Connect with vpn.uni-ulm.de and type in your Uni-Ulm kiz-Account-Name and kiz-Account-Password (use Cisco Secure Client for that). You will also be asked for your OTP from Uni Ulm (You can use Google Authenticator for     that).
4. Create a new session in MobaXterm with Session --> SSH Tab
   Type in the remote host: uc3.scc.kit.edu and your username (this is the username that was specified when you created the bw uni cluster account (for uni ulm it is ul_...). The port should be 22 by default, if not,         change it to 22. Click OK to create and save the session.
5. Login with the created session in MobaXterm (SSH). Just right click the created session and left click "Execute". 
   Again you will be asked to type in another OTP (OTP --> second factor for bw uni cluster) and then your service password for the BW Uni Cluster 3.0 you set when you created the account.
   You are now in your home-directory and this first section is completed.

## 2. Cluster Environment Preparation

1. The first step to be able to run a model on the bw uni cluster is to create a workspace, where you can store the model and your data. If you didn't yet register a worskapce use the following command in the terminal:
   - ws_register $HOME/workspaces
2. Next, you can allocate memory for your workspace with:
   - ws_allocate ProjectName 200 (replace ProjectName with your specific workspace/project name)
   - The number specifies the allocation size (200 means 200 GB) --> Change according to your needs
   - export WS_MODEL=$(ws_find ProjectName) --> (in our case ProjectName is pixtral)
3. We need to prepare the module and conda-environment (the following steps are used to load the pixtral model from mistral on the bw uni cluster, if you have another model, just adjust the names and paths):
   We use the following commands in the specified order:
   -  module load devel/miniforge/24.11.0-python-3.12
   -  source $(conda info --base)/etc/profile.d/conda.sh (load hook for conda)
   -  conda create -p $WS_MODEL/conda/pixtral python=3.10 -y
   -  conda activate $WS_MODEL/conda/pixtral
   - module load devel/cuda/12.8 (you can also first check which cuda versions are available with the following command: module avail cuda)
   - pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
   - pip install "vllm>=0.6.2" mistral_common>=1.4.4 pillow tqdm
You can evaluate if all the versions and modules are correct with the following code snippet: 
# Modul und Environment aktiviert
python - <<'PY'
import torch, subprocess, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime from PyTorch:", torch.version.cuda)
    print("GPU via nvidia-smi:")
    subprocess.run(["nvidia-smi", "-L"])
PY

4. Now we prepared the environment for the model and can pull the model pixtral-12b from huggingface
   - huggingface-cli login --> You need to create an access token on Hugging Face and type in this token as a login to clone the model on the bw uni cluster.
   - To create an access token you go to your profile on hugging face --> Access tokens --> Create new token --> Go to the tab Read --> Type in a meaningful token name --> Create Token
5. Load the pixtral model on the bw uni cluster with the following code:
   python - <<'PY'
   from huggingface_hub import snapshot_download
   import os, pathlib, sys
   os.environ["HF_HOME"] = os.getenv("WS_MODEL") + "/.hf_cache"

   snapshot_download(
       repo_id="mistralai/Pixtral-12B-Base-2409",
       allow_patterns=["*.safetensors","*.json","*.model"],
       local_dir=os.path.join(os.getenv("WS_MODEL"), "pixtral-12b"),
       local_dir_use_symlinks=False,
   )
   PY
   
7. The repository with the inference scripts for running models like pixtral has to be cloned.
   Therefore, you need to create an access token on github, because you need to clone the repository with SSH. Go to your profile (right corner) --> Settings --> Developer Settings (end of the page, scroll down) -->          Personal access tokens --> Tokens (classic) --> Generate new token --> Generate new token (classic)
   Type in a Note to describe roughly the purpose for the access token. Click on the scope repo (all things under repo should be checked) and click on "Generate token" at the bottom of the page. 
   Then type in the following command into the terminal: git@github.com:Wolfda95/MIRP_Benchmark_Student.git $WS_MODEL/mirp_benchmark
   You will be asked for your GitHub-Username and Password. 

## 3. Data & Directory Layout

Next, we load the custom data onto the bw uni cluster. If your data is relatively small, you can trasnfer it to your home-directory, otherwise, you should create another workspace for the data.
The most easy way to transfer the local data onto the bw uni cluster is to create a SFTP session in MobaXterm. New Session with right click in the session window --> SFTP Tab
Again type in the remote host just like in SSH: uc3.scc.kit.edu and type in your username. Use 22 as Port (should be the default setting). Click OK to create and save the session.
Then you can just drag and drop the dataset into the correct folder (for example a new directory in your home-directory with the name data).

## 4. Change paths in inference-script 

You have to change 3 main paths: One for the model, one for the data and one for the output of the results.
For the model path you have to change the following lines:




~/.bashrc.d/pixtral.sh
# 1) Verzeichnis anlegen (falls noch nicht da)
mkdir -p ~/.bashrc.d
# 2) Datei anlegen und öffnen
nano ~/.bashrc.d/pixtral.sh

MobaXterms Editor uses the Windows-Format for saving files. Change to unix.
scontrol show partition --> Command for devices available for jobs you run when running inference for the model
sbatch run_pixtral_mirp.sh to run Slurm-Script

Job-ID herausfinden: squeue -u $(whoami)
Job abbrechen: scancel 1234567

- Create a SFTP session for transfering the data to your cluster node.
- Access token for Hugging face --> Clone Pixtral-12b model
- Create Bash.rc for automating environemnt activation and to get into right directory
- install anaconda etc. all preparations necessary for running model
- transfer dataset (small) to bw uni cluster
- create slurm script
- run slurm script with changed directories in inference script for mistral
- run evaluation script with custom and qa jsons to compare outcome
- problems with slurm script: find correct gpu names etc.
- problems with directories in inference script: Now only one file instead of hierarchy --> same with evaluation scripts

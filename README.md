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
3. Connect with vpn.uni-ulm.de and type in your Uni-Ulm kiz-Account-Name and kiz-Account-Password (use Cisco Secure Client for that). You will also be asked for your OTP (You can use Google Authenticator for that).
4. Create a new session in MobaXterm with Session --> SSH
   Type in the remote host: uc3.scc.kit.edu and your username. The port should be 22 by default, if not, change it to 22.
5. Login with SSH (replace "yourUsername" with your actual Username you set up when you created the BW Uni Cluster 3.0 account: ssh yourUsername@uc3.scc.kit.edu
   Again you will be asked to type in another OTP and then your service password for the BW Uni Cluster 3.0 you set when you created the account.
   You are now in your home-directory and this first section is completed.

## 2. Cluster Environment Preparation




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

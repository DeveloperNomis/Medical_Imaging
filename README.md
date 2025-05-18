# Pipeline for setting up Mistral / Pixtral-12b on BW Uni Cluster 3.0 (Example setup)

This repository demonstrates how to set up an account, prepare an environment, run inference on simple “dot” spatial-relation tasks and evaluate results on the BW Uni Cluster 3.0 with a GPU.

## 📋 Table of Contents

1. [Account Setup](#account-setup)  
2. [Cluster Environment Preparation](#cluster-environment-preparation)  
3. [Data & Directory Layout](#data--directory-layout)  
4. [Inference Script & Slurm Submission](#inference-script--slurm-submission)  
5. [Evaluation Script](#evaluation-script)  
6. [Troubleshooting & Tips](#troubleshooting--tips)  


## 1. Account Setup

1. Follow this steps to create an account at BW Uni Cluster 3.0:  https://wiki.bwhpc.de/e/Registration/bwUniCluster
2. Download and install MobaXterm here: https://mobaxterm.mobatek.net/
3. Create a new session with Session --> SSH 
4. Connect with vpn.uni-ulm.de and type in your Uni-Ulm kiz-Account-Name and kiz-Account-Password (use Cisco Secure Client for that). You will also be asked for your OTP (You can use Google Authenticator for that).
5. Login with SSH (replace "yourUsername" with your actual Username you set up when you created the BW Uni Cluster 3.0 account: ssh yourUsername@uc3.scc.kit.edu
   Again you will be asked to type in another OTP and then your service password for the BW Uni Cluster 3.0 you set when you created the account.
6. 

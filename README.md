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
The used dataset has the following file structure:
alex_med-prompt
-->   images
         --> individual PNG-images
-->   question-answers.json

## 4. Change paths in inference-script 
Go to your inference script, with which you want to run the model. It should be in mirp_benchmark/inference_scripts/
Here it is done with the all_experiments_mistral.py and the above described dataset-structure (in section Data & Directory Layout). 
You have to change 3 main paths: One for the model, one for the data and one for the output of the results (according to your file structure) if you use the same file stucture for the data (Otherwise you have to further adjust the script, the original script uses 3 subfolders for example).
You have to change the following lines:
- model_dir
- dataset_dir
- RESULTS_ROOT

The script you can use is this one, just change the 3 lines and it should be working:

"""
Automated Image Processing and Model Call Script for Medical QA Tasks with Images

This script processes medical image datasets, selects images and corresponding 
question-answer (QA) data, and sends model calls to evaluate responses. 
It organizes the results and saves them as JSON files.

Workflow:
1. **Dataset and Task Selection**: 
   - The script defines multiple tasks (`experiments`) associated with different 
     image preprocessing techniques and QA files.
   - The dataset directory is set dynamically based on the selected dataset.

2. **Directory Setup**:
   - A results directory (`RESULTS_ROOT`) is created for each task.

3. **QA Data Extraction**:
   - QA data is read from JSON files for each task.
   - Images are randomly sampled or the full dataset is used.

4. **Image Processing**:
   - Images are converted to base64 format for model usage.

5. **Model Call Execution**:
   - A structured prompt is sent to the model with the image 
     and corresponding question.
   - Responses are collected and stored.

6. **Results Storage**:
   - Results are saved as JSON files with structured metadata.
   - Multiple runs are performed to validate consistency.

Dependencies:
    - `os`, `sys`, `json`, `random`, `time`, 
    - `torch`
    - `PIL` (for image processing)
    - `vllm`
    - `base64`
    - `io`

Notes:
    - Adjust dataset and task selection in `DATASETS` and `experiments`.
    - The script uses a fixed seed (`random.seed(2025)`) for reproducibility.
"""


import os
import sys
import json
import time
import random
import base64
from vllm import LLM
from vllm.sampling_params import SamplingParams
from io import BytesIO
from PIL import Image


def encode_image_from_bytes(image):
    """
    Encodes an image object into a base64-encoded PNG string.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        str: Base64-encoded string representation of the image.

    Example:
        ```python
        from PIL import Image
        import base64

        img = Image.open("example.png")
        encoded_string = encode_image_from_bytes(img)
        print(encoded_string)
        ```
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_grayscale(image):
    """
    Checks if a given image is in grayscale mode.

    Args:
        image (PIL.Image.Image): The image to check.

    Returns:
        bool: True if the image is grayscale, False otherwise.

    The function checks whether the image mode is either "L" (8-bit grayscale) 
    or "I;16" (16-bit grayscale).

    Example:
        ```python
        from PIL import Image

        img = Image.open("example.png")
        if is_grayscale(img):
            print("The image is grayscale.")
        else:
            print("The image is in color.")
        ```
    """
    return image.mode in ["L", "I;16"]


def normalize_16bit_to_8bit(image):
    """
    Normalizes a 16-bit grayscale image to an 8-bit grayscale image.

    Args:
        image (PIL.Image.Image): A 16-bit grayscale image (mode "I;16").

    Returns:
        PIL.Image.Image: An 8-bit grayscale image (mode "L").

    The function scales pixel values from the 16-bit range (0-65535) to the 
    8-bit range (0-255) by dividing each pixel by 256 and then converting 
    the image to mode "L".

    Example:
        ```python
        from PIL import Image

        img_16bit = Image.open("example_16bit.png")
        img_8bit = normalize_16bit_to_8bit(img_16bit)
        img_8bit.save("example_8bit.png")
        ```
    """
    normalized_image = image.point(
        lambda x: (x / 256))
    return normalized_image.convert("L")


def ensure_rgb(image):
    """
    Ensures that the given image is in RGB mode.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The image converted to RGB mode.

    This function handles different image modes as follows:
    - If the image is in "I;16" mode (16-bit grayscale), it is first normalized 
      to 8-bit grayscale and then converted to RGB.
    - If the image is grayscale ("L") or has an alpha channel ("RGBA"), 
      it is directly converted to RGB.
    - If the image is already in "RGB" mode, a copy is returned.
    - If the image mode is unsupported, a `ValueError` is raised.

    Example:
        ```python
        from PIL import Image

        img = Image.open("example.png")
        rgb_img = ensure_rgb(img)
        rgb_img.show()
        ```

    Raises:
        ValueError: If the image mode is not supported.
    """
    if image.mode == "I;16":
        return normalize_16bit_to_8bit(image).convert("RGB")
    elif is_grayscale(image) or image.mode == "RGBA":
        return image.convert("RGB")
    elif image.mode == "RGB":
        return image.copy()
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")


def get_clean_image(image_path):
    """
    Loads an image, ensures it is in RGB mode, and encodes it as a base64 string.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The base64-encoded representation of the image.

    This function performs the following steps:
    1. Opens the image from the given path.
    2. Converts it to RGB mode if necessary using `ensure_rgb()`.
    3. Encodes the processed image into a base64 string using `encode_image_from_bytes()`.

    Example:
        ```python
        encoded_image = get_clean_image("example.png")
        print(encoded_image)
        ```
    """
    with Image.open(image_path) as img:
        rgb_image = ensure_rgb(img)
    base64_image = encode_image_from_bytes(rgb_image)

    return base64_image


def get_qa(img_file_name, json_dir):
    """
    Retrieves the question-answer pairs for a given image file from a JSON dataset.

    Args:
        img_file_name (str): The filename of the image for which QA pairs are required.
        json_dir (str): The path to the JSON file containing question-answer data.

    Returns:
        list[dict]: A list of dictionaries, each containing a 'question' and an 'answer'.

    The function performs the following steps:
    1. Loads the JSON file from the provided directory.
    2. Finds the entry that matches the given image filename.
    3. Extracts and returns the associated question-answer pairs.

    Example:
        ```python
        qa_pairs = get_qa("image_001.jpg", "questions.json")
        for qa in qa_pairs:
            print(f"Q: {qa['question']}\nA: {qa['answer']}")
        ```
    """
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)

    target_filename = img_file_name
    result = next((entry['question_answer']
                   for entry in data if entry['filename'] == target_filename), None)

    questions_answers = [{'question': entry['question'],
                          'answer': entry['answer']} for entry in result]
    return questions_answers


def make_model_call(llm, questions_data, base64_image, additional_question):
    """
    Calls the model with a medical image and a question about its content.

    Args:
        llm (LLM): The loaded model.
        questions_data (dict): A dictionary containing:
            - 'question' (str): The question to ask about the image.
            - 'answer' (str): The expected answer.
        base64_image (str): The image base64-encoded.
        additional_question (dict): A dictionary containing:
            - 'question' (str): A sample question to demonstrate the response format.
            - 'answer' (str): The expected response format ('1' or '0').

    Returns:
        list[dict]: A list containing a single dictionary with:
            - 'question' (str): The question asked.
            - 'model_answer' (str): The cleaned AI-generated answer.
            - 'expected_answer' (str): The expected answer for comparison.
            - 'entire_prompt' (str): The full prompt used in the API call.

    The function constructs a strict yes/no prompt for the model, ensuring 
    a binary response ('1' for Yes, '0' for No). It sends the image in base64 
    format along with the textual question. The model response is then stored 
    along with the original question and expected answer.
    """
    # List for results
    results = []

    prompt = (
        "The image is a 2D axial slice of an abdominal CT scan with soft tissue windowing. "
        "Answer strictly with '1' for Yes or '0' for No. No explanations, no additional text. "
        "Your output must contain exactly one character: '1' or '0'."
        "Ignore anatomical correctness; focus solely on what the image shows.\n"
        "Example:\n"
        # dynamic part of the prompt
        f"Q: {additional_question['question']} A: {additional_question['answer']}\n"
        "Now answer the real question:\n\n"
        f"Q: {questions_data['question']}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    results.append({
        "question": questions_data['question'],
        "model_answer": outputs[0].outputs[0].text,
        "expected_answer": questions_data['answer'],
        "entire_prompt": prompt
    })

    return results


if __name__ == "__main__":

    model_dir = "/pfs/work9/workspace/scratch/ul_qmj57-pixtral/pixtral-12b/"

    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(
        model="/pfs/work9/workspace/scratch/ul_qmj57-pixtral/pixtral-12b/",
        tokenizer_mode="mistral",
        gpu_memory_utilization=0.95,  # Maximal GPU-Use
        max_model_len=32000,          # Reduced Sequence length
    )

    dataset_dir = "/home/ul/ul_student/ul_qmj57/data/alex_med-prompt"
    qa_file_path = os.path.join(dataset_dir, "qa.json")
    image_dir   = os.path.join(dataset_dir, "images")

    RESULTS_ROOT = os.path.join(model_dir, "results")  # path for results directory
    

    # experiments = ['RQ1', 'RQ2', 'RQ3', 'AS']  # select the experiments here
    
    # Load all QA-Pairs one time
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    # Create a Mapping with filename→question_answer
    qa_map = { entry['filename']: entry['question_answer']
               for entry in qa_data if 'filename' in entry }

    # All image filenames in images/
    png_images = sorted(os.listdir(image_dir))
    random.seed(2025)

    # Choose, how many images you want to test randomly,
    # or just use N = len(png_images), to take all images:
    N = len(png_images)
    if N < len(png_images):
        png_images = random.sample(png_images, N)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Single „Experiment“ – n Runs
    for run_idx in range(3):
        results = []
        start = time.time()

        for img in png_images:
            # Question-Answer-Pair for this image
            qa_pairs = qa_map.get(img)
            if not qa_pairs:
                print(f"Warnung: keine QA für {img}")
                continue

            # Example question for Prompt-Format
            additional = qa_pairs[0] if len(qa_pairs)>1 else qa_pairs[0]

            img_path = os.path.join(image_dir, img)
            b64 = get_clean_image(img_path)

            # Only first QA-Pair is used;
            # you can also iterate through all qa_pairs
            result = make_model_call(llm, qa_pairs[0], b64, additional)[0]
            results.append({"file": img, "result": result})

        elapsed = time.time() - start
        out_name = f"custom_all_{N}_imgs_run{run_idx}.json"
        with open(os.path.join(RESULTS_ROOT, out_name), 'w') as out:
            json.dump(results, out, indent=2)

        print(f"Run {run_idx} with {N} images finished in {elapsed:.1f}s.")


## 5. Slurm Submission and starting inference with model
To start a run with our model on the dataset, we need to first create a slurm script, that automatically initializes the environment, references all paths etc.
First type into the terminal: nano run_pixtral_mirp.sh (for example, you can of course choose another name).
Use this code as a basis and change it according to your paths and need for devices:

```bash
#!/usr/bin/env bash
#SBATCH --partition=dev_gpu_h100        # GPU-Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 00:20:00                     # 20 Min Testlauf
#SBATCH -J pixtral_gpu_dev              # Job-Name
#SBATCH --output=slurm_log/pixtral_gpu.%j.out

## 0) Workspace & Pfade ------------------------------------------------
export WS_MODEL=$(ws_find pixtral)
export DATA_DIR="$HOME/data/alex_med-prompt"
export RESULTS_DIR="$WS_MODEL/pixtral-12b/results"
mkdir -p "$RESULTS_DIR" slurm_log

## 1) Miniforge laden & Env-Python in PATH -----------------------------
module load devel/miniforge/24.11.0-python-3.12
# kein `conda activate` => vermeiden wir den CLI-Import-Error
export PATH="$WS_MODEL/conda/pixtral/bin:$PATH"

## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`

## 3) Kurz-Check (ASCII-nur, ohne Umlaut) ------------------------------
python - <<'PY'
import importlib, vllm, torch
print("vllm im Pfad:", bool(importlib.util.find_spec("vllm")))
print("CUDA available:", torch.cuda.is_available())
PY

## 4) Inferenz ---------------------------------------------------------
python "$WS_MODEL/mirp_benchmark/inference_scripts/all_experiments_mistral.py" \
       --model_path "$WS_MODEL/pixtral-12b" \
       --data_path  "$DATA_DIR/images" \
       --output_dir "$RESULTS_DIR" \
       --batch_size 4
```




## 6. Automation of environment activation and variables

To automatize the whole procedure with activating the conda environment, loading the necessary modules, set environment-variables and setting the directory, we create a script that does that automatically when logging    onto your bw uni cluster account.
   - Therefore, we first need to add this code block to our .bashrc in our home-directory:
   if [ -d ~/.bashrc.d ]; then
	for rc in ~/.bashrc.d/*; do
		if [ -f "$rc" ]; then
			. "$rc"
		fi
	done
fi

With this additional block we can create a new directory (if not already created):
- mkdir -p ~/.bashrc.d
And then create a file in there named pixtral.sh (for example).
- nano ~/.bashrc.d/pixtral.sh
Put the following code into this file:
#!/usr/bin/env bash
# === Pixtral-Setup für interaktive Shells ===
if [[ $- == *i* ]] && ws_find pixtral &> /dev/null; then
  export WS_MODEL=$(ws_find pixtral)
  export DATA_DIR="/home/ul/ul_student/ul_qmj57/data/alex_med-prompt"
  export RESULTS_DIR="$WS_MODEL/results"

  module load devel/miniforge/24.11.0-python-3.12
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$WS_MODEL/conda/pixtral"
  module load devel/cuda/12.8

  cd "$WS_MODEL" || true
fi
# === Ende Pixtral-Setup ===

Of course, the paths need to changed accordingly to your paths. Save the file and exit.
Now, every time you log onto the bw uni cluster, all needed environments, variables etc. are correctly initialized.



MobaXterms Editor uses the Windows-Format for saving files. Change to unix.
scontrol show partition --> Command for devices available for jobs you run when running inference for the model
sbatch run_pixtral_mirp.sh to run Slurm-Script

Job-ID herausfinden: squeue -u $(whoami)
Job abbrechen: scancel 1234567

- run slurm script with changed directories in inference script for mistral
- problems with slurm script: find correct gpu names etc.

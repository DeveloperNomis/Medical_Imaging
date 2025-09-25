# Pipeline for setting up Mistral / Pixtral-12b on BW Uni Cluster 3.0 (Example setup)

This repository demonstrates how to set up an account, prepare an environment and run inference on a dataset with the pixtral model on the BW Uni Cluster 3.0 with a GPU.  
It also includes scripts for structured parallel experiment execution via SLURM array jobs and for aggregating/evaluating results.

## 📋 Table of Contents

1. [Account Setup and Login](#account-setup)  
2. [Cluster Environment Preparation](#cluster-environment-preparation)  
3. [Data](#data)  
4. [Change paths in inference-script](#change-paths--inference-script)  
5. [Slurm Submission and starting inference with model](#Slurm-Submission)  
6. [Automation of environment activation and variables](#automation)
7. [Parallel processing of prompts](#parallel-processing)


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
```python
# Activated module and environment
python - <<'PY'
import torch, subprocess, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime from PyTorch:", torch.version.cuda)
    print("GPU via nvidia-smi:")
    subprocess.run(["nvidia-smi", "-L"])
PY
```

4. Now we prepared the environment for the model and can pull the model pixtral-12b from huggingface
   - huggingface-cli login --> You need to create an access token on Hugging Face and type in this token as a login to clone the model on the bw uni cluster.
   - To create an access token you go to your profile on hugging face --> Access tokens --> Create new token --> Go to the tab Read --> Type in a meaningful token name --> Create Token
5. Load the pixtral model on the bw uni cluster with the following code:
```python
   python - <<'PY'
   from huggingface_hub import snapshot_download
   import os, pathlib, sys
   os.environ["HF_HOME"] = os.getenv("WS_MODEL") + "/.hf_cache"

   snapshot_download(
       repo_id="mistralai/Pixtral-12B-2409",
       allow_patterns=["*.safetensors","*.json","*.model"],
       local_dir=os.path.join(os.getenv("WS_MODEL"), "pixtral-12b"),
       local_dir_use_symlinks=False,
   )
   PY
```
7. The repository with the inference scripts for running models like pixtral has to be cloned.
   Therefore, you need to create an access token on github, because you need to clone the repository with SSH. Go to your profile (right corner) --> Settings --> Developer Settings (end of the page, scroll down) -->          Personal access tokens --> Tokens (classic) --> Generate new token --> Generate new token (classic)
   Type in a Note to describe roughly the purpose for the access token. Click on the scope repo (all things under repo should be checked) and click on "Generate token" at the bottom of the page. 
   Then type in the following command into the terminal: git@github.com:Wolfda95/MIRP_Benchmark_Student.git $WS_MODEL/mirp_benchmark
   You will be asked for your GitHub-Username and Password.

## 3. Data

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
Go to your inference script, with which you want to run the model. It should be in mirp_benchmark/inference_scripts/ (You can open any script with the MobaTextEditor). Just make sure to change the format to unix: Format in the above bar --> Unix and if necessary the Encoding to UTF-8 if you use umlauts in comments or something similar (Encoding in the above bar --> UTF-8 (default))
Here it is done with the all_experiments_mistral.py and the above described dataset-structure (in section Data & Directory Layout). 
You have to change 3 main paths: One for the model, one for the data and one for the output of the results (according to your file structure) if you use the same file stucture for the data (Otherwise you have to further adjust the script, the original script uses 3 subfolders for example).
You have to change the following lines:
- model_dir
- dataset_dir
- RESULTS_ROOT

The script you can use is this one, just change the 3 lines and it should be working:
```python
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
```


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

If you want to use a different device, you can check the available devices with the following command in the terminal in your workspace:
- scontrol show partition
This command lists the devices available for jobs you run (with the above slurm script) when running inference for the model.
But you have to be careful, for different devices, not only the line with "--partition" changes, but you have to adjust some other lines accordingly.

To finally run the slurm script, type in the following command:
- sbatch run_pixtral_mirp.sh
You can then list all running jobs and their ids:
- squeue -u $(whoami)
And also cancel a running job:
- scancel 1234567 (Of course you have to change the number to the specific job id you want to cancel)

This setup runs the inference script for the pixtral model and a specific dataset on the bw uni cluster.
Just be careful that all paths exist that you refer to, then this setup should be working fine.
And remember, that you need to extend the time for the workspace if you need it longer, for example:
- ws resize pixtral --lifetime 30 (extends the workspace time for 30 days)
You can also check the remaining time with:
- ws info pixtral
If you want to have the results from your pixtral model permanently, you can copy them into your home directory:
- cp -r "$WS_MODEL/pixtral-12b/results" "$HOME/pixtral_results"
Of course, the directory pixtral_results has to exist in your home-directory.

## 6. Automation of environment activation and variables

To automatize the whole procedure with activating the conda environment, loading the necessary modules, set environment-variables and setting the directory, we create a script that does that automatically when logging    onto your bw uni cluster account.
   - Therefore, we first need to add this code block to our .bashrc in our home-directory:
```bash
   if [ -d ~/.bashrc.d ]; then
	for rc in ~/.bashrc.d/*; do
		if [ -f "$rc" ]; then
			. "$rc"
		fi
	done
fi
```
With this additional block we can create a new directory (if not already created):
- mkdir -p ~/.bashrc.d
And then create a file in there named pixtral.sh (for example).
- nano ~/.bashrc.d/pixtral.sh
Put the following code into this file:
```bash
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
```
Of course, the paths need to changed accordingly to your paths. Save the file and exit.
Now, every time you log onto the bw uni cluster, all needed environments, variables etc. are correctly initialized.

## 7. Parallel processing of prompts
So far, we manually changed the prompt in the inference script and every new prompt was processed on its own.
We now would like to parallely process several prompts.The idea is to automatically read in a text file with all the prompts and the main slurm-script calls the inference-script for each prompt individually. 
The inference script then parses the given text and replaces the prompt with this information by which the Pixtral-12B model is called.
Then we have multiple parallel jobs which create a number of result directories. These result directories are saved by a unique hash id. 
By calling an evaluation script, one can derive metrics from the results and visualize them in plots (we get to this again later).
Missing: Parsing for inference script (code), Slurm-Scirpt which executes several prompts parallely (code), Prompts-Text file (one quick example), Evaluation script, Visualization code, command to execute slurm script.  

The text file should look like this, where each prompt is in its own line:    
```txt
Prompt 1
Prompt 2
Prompt 3
...
```

```bash
#!/usr/bin/env bash
###############################################################################
# pixtral_array.slurm  – 1 Prompt pro Array-Task, verarbeitet RQ1-3 (+ Marker)
###############################################################################

############################  Ressourcen  #####################################
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH -J pixtral_h100
#SBATCH --output=slurm_log/pixtral_%A_%a.out
#SBATCH --error=slurm_log/pixtral_%A_%a.err
###############################################################################

# ---------------------------------------------------------------------------
# 0) Stub „conda“ unschädlich machen, *dann* strenges Bash-Profil aktivieren
# ---------------------------------------------------------------------------
alias conda="true"                 # vermeidet ImportError & Exit-Code≠0

# ---------------------------------------------------------------------------
# 1) Grundpfade (WS_MODEL *vor* PROMPTS_FILE setzen!)
# ---------------------------------------------------------------------------
export WS_MODEL=$(ws_find pixtral)   # Cluster-Helper

PROMPTS_FILE="${1:-$WS_MODEL/mirp_benchmark/inference_scripts/prompts/prompts.txt}"
export DATA_DIR="$HOME/data/MISR_Dataset"
export RESULTS_ROOT="$WS_MODEL/pixtral-12b-finetuned/results"   # Wurzel für alle Runs

mkdir -p "$RESULTS_ROOT" slurm_log

# ---------------------------------------------------------------------------
# 2) Prompt für diesen Array-Task holen
# ---------------------------------------------------------------------------
PROMPT_TEXT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$PROMPTS_FILE")
[[ -z "$PROMPT_TEXT" ]] && { echo "Leere Prompt-Zeile – Task endet."; exit 0; }

PROMPT_HASH=$(echo -n "$PROMPT_TEXT" | md5sum | cut -d' ' -f1)
PROMPT_OUTDIR="$RESULTS_ROOT/$PROMPT_HASH"
mkdir -p "$PROMPT_OUTDIR"
echo "$PROMPT_TEXT" > "$PROMPT_OUTDIR/PROMPT.txt"

# ---------------------------------------------------------------------------
# 3) Module & Python-Umgebung
# ---------------------------------------------------------------------------
module load devel/miniforge/24.11.0-python-3.12
export PATH="$WS_MODEL/conda/pixtral/bin:$PATH"
module load devel/cuda/12.8

set -euo pipefail

# ---------------------------------------------------------------------------
# 4) Temp-Datei für --prompt_file
# ---------------------------------------------------------------------------
TMP_PROMPT=$(mktemp /tmp/prompt_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXX.txt)
echo "$PROMPT_TEXT" > "$TMP_PROMPT"
trap 'rm -f "$TMP_PROMPT"' EXIT      # automatische Aufräum­routine

# ---------------------------------------------------------------------------
# 5) Inferenz – Py-Script erzeugt Unterordner pro RQ & Marker
# ---------------------------------------------------------------------------
python "$WS_MODEL/mirp_benchmark/inference_scripts/all_experiments_mistral_dynamic.py" \
       --model_path  "$WS_MODEL/pixtral-12b-finetuned" \
       --data_path   "$DATA_DIR" \
       --output_root "$PROMPT_OUTDIR" \
       --batch_size  4 \
       --prompt_file "$TMP_PROMPT"

echo "Task $SLURM_ARRAY_TASK_ID abgeschlossen – Ergebnisse unter $PROMPT_OUTDIR"
```

In general, the script looks similar to the first inference-script.  
This script is the production-ready replacement for the earlier development job file.
It is designed for large-scale prompt-based inference on the cluster.

Key features:
  - Runs as an array job with specific command: each array task processes exactly one prompt from a prompt file (prompts.txt in this case)
  - Uses the SLURM environment variable SLURM_ARRAY_TASK_ID to select the correct prompt line
  - Creates a unique hash-based output directory per prompt, and stores the prompt text alongside the results
  - Requests 1 H100 GPU, 8 CPUs, 64 GB RAM, up to 30 hours runtime (--partition=gpu_h100)
  - Uses a temporary file for the prompt (--prompt_file) with automatic cleanup via trap
  - Logs are stored separately per task:
  	 - slurm_log/pixtral_%A_%a.out
    - slurm_log/pixtral_%A_%a.err
  - Loads miniforge (Python 3.12) and CUDA 12.8, avoids conda activate issues by directly exporting the env path
  - Runs inference via
```bash
python .../all_experiments_mistral_dynamic.py \
    --model_path pixtral-12b-finetuned \
    --data_path <DATA_DIR> \
    --output_root <PROMPT_OUTDIR> \
    --batch_size 4 \
    --prompt_file <TMP_PROMPT>
```
Differences from the old script:
  - Old script was a single short test job (20 min limit, dev queue) with a simple check for CUDA + vLLM
  - New script supports long runs (30h) and scales to many prompts via job arrays
  - Old script used a fixed dataset path and single output folder, new script organizes results per prompt
  - New script has robust error handling (set -euo pipefail, empty prompt check, trap cleanup)

How to run:
```bash
sbatch --array=0-9 run_pixtral_mirp_all_dynamic.sh prompts.txt
```
This would process the first 10 prompts from prompts.txt.  
Of course one needs to specify the correct directories and change the names of the files accordingly.  

Results will be stored under:  
```bash
$WS_MODEL/pixtral-12b-finetuned/results/<PROMPT_HASH>/
```
with the corresponding PROMPT.txt saved for reproducibility.  

```python

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
from string import Template
import argparse



def load_prompt_template(path: str) -> Template:
    with open(path, encoding="utf-8") as f:
        return Template(f.read())


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


def get_clean_image(original_image_path):
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
    with Image.open(original_image_path) as img:
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

    prompt = PROMPT_TEMPLATE.substitute(
        example_question=additional_question['question'],
        example_answer=additional_question['answer'],
        question=questions_data['question']
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



    # ---- 1. CLI-Argumente einlesen -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default="prompts/promptsV1.txt",
                        help="Pfad zur Prompt-Vorlage")
                        
    parser.add_argument("--model_path", required=True,
                    help="Ordner mit dem finetuned Modell")
    parser.add_argument("--data_path", required=True,
                    help="Wurzelverzeichnis des MISR-Datasets")
    parser.add_argument("--output_root", required=True,
                    help="Wohin die Ergebnis-JSONs geschrieben werden")
    parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch-Größe für vllm (optional)")
                    
    args = parser.parse_args()
    
    # ---- 2. Prompt-Template laden --------------------------
    PROMPT_TEMPLATE = load_prompt_template(args.prompt_file)
    
    

    model_dir = args.model_path

    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(
        model=args.model_path,
        tokenizer_mode="mistral",
        gpu_memory_utilization=0.95,  # Maximale GPU-Nutzung
        max_model_len=32000,          # Reduzierte Sequenzlänge
    )

    dataset_dir = args.data_path

    RESULTS_ROOT = args.output_root

    experiments = ['RQ1', 'RQ2', 'RQ3']  # select the experiments here

    for exp in experiments:

        if exp == 'RQ1':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images',
                                     'qa': 'qa.json'}
            }
            
        elif exp == 'RQ2':
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers',
                                     'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters',
                                     'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',
                                     'qa': 'qa_dots.json'}
            }

        else:
            experiment_plan = {
                'sub_experiment_1': {'img': 'images_numbers',
                                     'qa': 'qa_numbers.json'},
                'sub_experiment_2': {'img': 'images_letters',
                                     'qa': 'qa_letters.json'},
                'sub_experiment_3': {'img': 'images_dots',
                                     'qa': 'qa_dots.json'}
            }

        exp_dir = os.path.join(dataset_dir, exp)

        for sub_experiment, data in experiment_plan.items():

            selected_image = data['img']
            selected_qa = data['qa']

            qa_file_path = os.path.join(exp_dir, selected_qa)

            image_files_path = os.path.join(exp_dir, selected_image)

            with open(qa_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            png_images = [entry['filename']
                          for entry in data if 'filename' in entry]

            random.seed(2025)
            
            target_dir = os.path.join(RESULTS_ROOT, exp, sub_experiment)
            os.makedirs(target_dir, exist_ok=True)

            N = len(png_images)  # number or len(png_images)

            if N > len(png_images):
                print(f'The selected amount of images {N} is bigger than the available images {len(png_images)}.')
                sys.exit(0)
            elif N == len(png_images):
                print(f'The selected amount of images {N} is equal to the available images {len(png_images)}. Not picking random, using whole dataset instead.')
                mo_file_name_appendix = 'all_images'
                png_images = png_images
            else:
                print(f'Using random pick with {N} images.')
                png_images = random.sample(png_images, N)
                mo_file_name_appendix = f'random_pick_{N}_images'

            for i in range(5):  # how many runs ?
                start_time = time.time()

                dataset_results = []

                for image in png_images:
                    question_data = get_qa(image, qa_file_path)

                    other_images = [
                        img for img in png_images if img != image]
                    if other_images:
                        random_other_image = random.choice(other_images)
                        additional_question = get_qa(
                            random_other_image, qa_file_path)
                    else:
                        additional_question = None

                    original_image_path = os.path.join(
                        image_files_path, image)

                    base64_image = get_clean_image(original_image_path)

                    results_call = make_model_call(llm,
                                                   question_data[0], base64_image,
                                                   additional_question=additional_question[0])

                    dataset_results.append({
                        "file_name": image,
                        "results_call": results_call
                    })

                results_file_name = f"{selected_qa.replace('.json', '')}_{mo_file_name_appendix}_add_run_{i}.json"

                save_name = os.path.join(
                    target_dir, results_file_name)

                with open(save_name, 'w') as json_file:
                    json.dump(dataset_results, json_file, indent=4)
                end_time = time.time()

                elapsed_time = end_time - start_time
                print(f"Runtime for {selected_qa.replace('.json', '')} with {selected_image} : {elapsed_time:.2f} seconds")
```

As you probably already have seen, this is the second version of the python inference script for medical image QA evaluation using VLLM.  
The new script (recommended) improves flexibility, reproducibility, and experiment organization.  

### Key Improvements in the New Script:  
  - Command-line interface with arguments:
	--model_path, --data_path, --output_root, --prompt_file, --batch_size
	→ No more hard-coded paths
  - Prompt templates: Instead of hard-coded strings, the prompt is loaded from an external file (prompts/*.txt) and filled dynamically
  - Experiment structure: Supports multiple experiments (RQ1, RQ2, RQ3) with sub-experiments and separate QA/image sets
  - Organized outputs: Results are stored per experiment and sub-experiment under the chosen --output_root directory
  - Multiple runs: By default, each sub-experiment is repeated 5 times to check consistency
  - Additional question handling: Uses a random QA-pair from another image to enrich the prompt format
  - Note: --batch_size argument is parsed but not yet used in the inference call.  

If you now execute the slurm script script with the command specified above, you get all the different results for the prompts.
Next, we want to evaluate the results. This means we compute metrics from it to really compare the performance.
The evaluation script can be found in the directory "Evaluation" in this repository. 

### What you might need to change
#### Base paths: 
  - In main(): base_path = os.path.join(os.environ["WS_MODEL"], "pixtral-12b-finetuned", "results")
  -	unsure_base = os.path.join(os.environ["WS_MODEL"], "pixtral-12b-finetuned", "unsure_cases")
  -	If your model/results live elsewhere, adjust these paths (or set WS_MODEL accordingly)
#### Folder names / ordering:
  - Excel/CSV sorting prioritizes RQ1, RQ2, RQ3, AS1, AS2. Change the priority map if your naming differs.
#### Number of runs in the summary table:
  - The sheet reserves columns for 5 runs. If you use more/less, tweak N_RUNS = 5 in the Excel/CSV writers.

### Outputs
  - Excel: ${base_path}/Results_Images.xlsx
  - CSV: ${base_path}/Results_Images.csv
  - Summary JSONs in unsure_cases/...
The CSV/Excel Files contain only the aggregated metrics (Accuracy, F1, mean/std, counts).  
The Combined JSON files additionally include per-run details and the full list of unsure cases (unparseable answers).  
You run this python script by just typing this command:
```bash
python3 FiveRuns_1_calculate_results_image_full.py
```

### CSV Summary Contents
The generated Results_Images.csv provides an overview of all experiments.  
Each row corresponds to one <RQ>/<marker>/<variant> folder and includes:  
  - Aggregated metrics:
    - Accuracy_Mean, Accuracy_Std
    - F1_Mean, F1_Std
    - subset metrics for Left/Right and Above/Below questions (mean/std)
  - Run-level counts:
    - correct_run1…N, incorrect_run1…N, unsure_run1…N
This file is intended for high-level comparison across experiments.  
Detailed per-run results and unsure cases are only available in the combined JSON files.


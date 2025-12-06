import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CrackSegDiff Training and Testing on Colab\n",
    "\n",
    "This notebook sets up the environment, downloads data and weights, trains the CrackSegDiff model, and runs inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Check GPU\n",
    "!nvidia-smi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Clone Repository (if not already present)\n",
    "import os\n",
    "if not os.path.exists('CrackSegDiff'):\n",
    "    !git clone https://github.com/Ludwig-H/CrackSegDiff.git\n",
    "%cd CrackSegDiff\n",
    "!git pull"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Install Dependencies\n",
    "!pip install -r requirement.txt\n",
    "!pip install causal_conv1d>=1.0.0"
    "!pip install mamba_ssm>=1.0.1"

    "!pip install zenodo_get"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Prepare Pretrained Weights\n",
    "!mkdir -p pretrained_weights\n",
    "# Download from Google Drive using gdown\n",
    "!gdown 1JYqMxM5dbCLZ-WGPKtIofYJhj0VPuy3l -O pretrained_weights/vssm_base_0229_ckpt_epoch_237.pth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Patch Code (Fix hardcoded paths)\n",
    "import os\n",
    "\n",
    "target_file = 'CrackSegDiff/guided_diffusion/unet.py'\n",
    "old_path = '/home/dell/jlc/segdiff/pre_trained_weights/vssm_base_0229_ckpt_epoch_237.pth'\n",
    "new_path = os.path.abspath('pretrained_weights/vssm_base_0229_ckpt_epoch_237.pth')\n",
    "\n",
    "if os.path.exists(target_file):\n",
    "    with open(target_file, 'r') as f:\n",
    "        content = f.read()\n",
    "    \n",
    "    if old_path in content:\n",
    "        print(f\"Patching {target_file}...\")\n",
    "        content = content.replace(old_path, new_path)\n",
    "        with open(target_file, 'w') as f:\n",
    "            f.write(content)\n",
    "        print(\"Success!\")\n",
    "    else:\n",
    "        print(\"Path not found or already patched.\")\n",
    "else:\n",
    "    print(f\"File {target_file} not found. Check PWD.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Prepare Dataset\n",
    "!mkdir -p data\n",
    "%cd data\n",
    "!zenodo_get 6383044\n",
    "!unzip -q -o *.zip\n",
    "%cd ..\n",
    "\n",
    "# Organize data structure for CustomDataset loader\n",
    "# Expects: data_dir/5d/*.png and data_dir/mask/*.bmp\n",
    "import os\n",
    "import shutil\n",
    "import glob\n",
    "\n",
    "# Define where we want our training and test sets\n",
    "base_data_dir = os.path.abspath('data')\n",
    "train_dir = os.path.join(base_data_dir, 'train_formatted')\n",
    "test_dir = os.path.join(base_data_dir, 'test_formatted')\n",
    "\n",
    "for d in [train_dir, test_dir]:\n",
    "    os.makedirs(os.path.join(d, '5d'), exist_ok=True)\n",
    "    os.makedirs(os.path.join(d, 'mask'), exist_ok=True)\n",
    "\n",
    "print(\"Organizing files... this might take a moment.\")\n",
    "\n",
    "# Heuristic: Find all .png images in 'data'. \n",
    "# If FIND dataset has 'train' and 'test' folders, we use them.\n",
    "# Otherwise, we might need to split manually. \n",
    "# Let's look for known folder names from FIND dataset zip extraction.\n",
    "# Usually it extracts to something like 'FIND Dataset' folder.\n",
    "\n",
    "found_dirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]\n",
    "print(f\"Found directories in data: {found_dirs}\")\n",
    "\n",
    "# Attempt to find images recursively\n",
    "all_images = glob.glob('data/**/*.png', recursive=True)\n",
    "all_masks = glob.glob('data/**/*.bmp', recursive=True)\n",
    "\n",
    "print(f\"Found {len(all_images)} images and {len(all_masks)} masks.\")\n",
    "\n",
    "# Simple 80/20 split if no explicit train/test folders found\n",
    "# Note: This is a fallback. If the zip has structure, we should use it.\n",
    "# Assuming filenames match for image and mask (except extension)\n",
    "\n",
    "pairs = []\n",
    "for img_path in all_images:\n",
    "    # construct expected mask path\n",
    "    # This logic depends heavily on dataset naming convention. \n",
    "    # Checking if a corresponding .bmp exists with same basename\n",
    "    basename = os.path.splitext(os.path.basename(img_path))[0]\n",
    "    # Look for mask with same basename\n",
    "    matching_masks = [m for m in all_masks if os.path.splitext(os.path.basename(m))[0] == basename]\n",
    "    if matching_masks:\n",
    "        pairs.append((img_path, matching_masks[0]))\n",
    "\n",
    "print(f\"Matched {len(pairs)} image-mask pairs.\")\n",
    "\n",
    "split_idx = int(len(pairs) * 0.8)\n",
    "train_pairs = pairs[:split_idx]\n",
    "test_pairs = pairs[split_idx:]\n",
    "\n",
    "def copy_data(pair_list, dest_dir):\n",
    "    for img, mask in pair_list:\n",
    "        shutil.copy(img, os.path.join(dest_dir, '5d'))\n",
    "        shutil.copy(mask, os.path.join(dest_dir, 'mask'))\n",
    "\n",
    "if len(pairs) > 0:\n",
    "    copy_data(train_pairs, train_dir)\n",
    "    copy_data(test_pairs, test_dir)\n",
    "    print(\"Data organized successfully.\")\n",
    "else:\n",
    "    print(\"ERROR: No data pairs found. Please check dataset structure.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. Train\n",
    "import os\n",
    "data_dir = os.path.abspath('./data/train_formatted')\n",
    "out_dir = os.path.abspath('./results/train_output')\n",
    "\n",
    "# Ensure directories exist\n",
    "os.makedirs(out_dir, exist_ok=True)\n",
    "\n",
    "print(f\"Training with data from: {data_dir}\")\n",
    "\n",
    "!python CrackSegDiff/segmentation_train.py --data_dir {data_dir} --out_dir {out_dir} --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 8 --save_interval 5000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8. Inference\n",
    "import glob\n",
    "\n",
    "# Get latest model\n",
    "models = sorted(glob.glob('./results/train_output/*.pt'))\n",
    "if not models:\n",
    "    print(\"No trained model found. Using pretrained if available.\")\n",
    "    model_path = \"./pretrained_weights/savedmodel100000.pt\" # Fallback path, might need adjustment if you want to use the one you downloaded\n",
    "else:\n",
    "    model_path = models[-1]\n",
    "\n",
    "print(f\"Using model: {model_path}\")\n",
    "\n",
    "test_data_dir = os.path.abspath('./data/test_formatted')\n",
    "\n",
    "for modality in ['intensity', 'range', 'fused', 'filtered']:\n",
    "    out_path = f\"./results/test_output_{modality}\"\n",
    "    os.makedirs(out_path, exist_ok=True)\n",
    "    print(f\"Testing {modality}...\")\n",
    "    !python CrackSegDiff/segmentation_sample.py --data_dir {test_data_dir} --out_dir {out_path} --model_path {model_path} --modality {modality} --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 1\n",
    "    print(f\"Done {modality}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9. Archive results\n",
    "!zip -r results.zip ./results"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("CrackSegDiff_Colab.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)

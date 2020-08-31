#!/bin/bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --time=5-12
#SBATCH --mem=80000
#SBATCH --partition=docker
#SBATCH --account=docker

# Include in the header for email updates: 
#  #SBATCH --mail-type=ALL
#  #SBATCH --mail-user=john.doe@biginternetcompany.com

# Define paths here or modify them in the command line arguments below
$INPUT_PATH=""
$WEIGHTS_PATH=""
$OUTPUT_PATH=""

# Some semi-fixed paths
$UNET_SCRIPT_PATH="/mnt/focus/Caplan/Scripts/Stromules/segmentation/scripts/unet_seg.py"

# Need to run docker commands with "sudo" and omit the "--gpus all" flag for the docker partition
sudo docker run \
-v /mnt/focus:/mnt/focus unet_container \
python3 -u $UNET_SCRIPT_PATH \
--seed 1337 \
--mode predict \
--input-path $INPUT_PATH \
--seg-channels 0 \
--image-ext .tif \
--weights-path $WEIGHTS_FILE \
--output-path $OUTPUT_PATH \
--num-classes 2 \
--output-ext .tif \
--checkpoint-directory None \
--epochs 50 \
--batch-size 256 \
--patch-size 64 \
--patch-thresh 0.05 \
--mask-ext .tif \
--workflow-type segmentation

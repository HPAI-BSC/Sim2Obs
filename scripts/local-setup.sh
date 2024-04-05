#!/bin/bash
# Create local folder structure and download data from minio
# The dataset consist on images and the label is embedded in the image name:
#      Example: idx-12_label-5.png, idx-<index in the dataset>_label-<label of the image>.png
# Run from root folder with: bash scripts/local-setup.sh

mkdir -p local/data
mkdir -p local/output

echo "Download data from minio and put it in local/data/"

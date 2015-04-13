#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l mem=32GB
#PBS -l walltime=2:00:00
#PBS -N DEEP_BLUE
 
module purge
module load torch-deps/7

# Define path to torch
alias th=/home/erc399/torch/install/bin/th

# Make team directory
cd $HOME
mkdir DEEP_BLUE
cd DEEP_BLUE

# Define download URL
url=http://cims.nyu.edu/~erc399/DL_DeepBlue_a3_YDS/submission/

# Download files
# Get skeleton
wget ${url}DEEP_BLUE_A3_skeleton.lua

# Get baseline
wget ${url}DEEP_BLUE_A3_baseline.lua

# Get model
wget ${url}DEEP_BLUE_model.sh
wget ${url}DEEP_BLUE_model.lua

# Get paper
wget ${url}DEEP_BLUE_model.pdf


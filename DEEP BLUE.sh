#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l mem=32GB
#PBS -l walltime=2:00:00
#PBS -N DEEP BLUE
 
module purge
module load torch-deps/7

# Define path to torch
alias th=/home/erc399/torch/install/bin/th

# Make team directory
cd $HOME
mkdir DEEP\ BLUE
cd DEEP\ BLUE

# Define download URL
url=http://cims.nyu.edu/~erc399/DL_DeepBlue_a3_YDS/submission/

# Download files
# Get skeleton
wget ${url}DEEP\ BLUE_A3_skeleton.lua

# Get baseline
wget ${url}DEEP\ BLUE_A3_baseline.lua

# Get model
wget ${url}DEEP\ BLUE_model.sh
wget ${url}DEEP\ BLUE_model.net

# Get paper
wget ${url}DEEP\ BLUE_model.pdf
#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /scratch-shared/dwu18/cache/logs/out.calibration.%j.o
#SBATCH -o /scratch-shared/dwu18/cache/logs/out.calibration.%j.e

source activate calibration

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/home/dwu18/anaconda3/envs/calibration

python inference_vllm.py

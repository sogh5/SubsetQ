# Subset Quantization

Implementation for Non-Uniform Step Size Quantization for Accurate Post-Training Quantization (ECCV 2022)


## Requirements
 - Python v3.6
 - PyTorch 1.7.1
 - CUDA 11.1 with cuDNN 8 (We use NVIDIA GeForce RTX 3090 GPU)

## How to Run Subset Quantization
	./run_ptq.sh [DATASET_PATH] [WBIT] [GPU_ID]

## How to Run Evaluation
	./test_ptq.sh [DATASET_PATH] [MODEL_FILE_PATH] [GPU_ID]

## etc.
 - Currently, we support only single GPU mode.
 - Our codes are based on implementation: https://github.com/yhhhli/BRECQ (Copyright (c) 2021 Yuhang Li, under MIT License).
   - Please refer to the files: [NOTICE.md](NOTICE.md), [BRECQ-LICENSE](ex_lics/BRECQ-LICENSE)

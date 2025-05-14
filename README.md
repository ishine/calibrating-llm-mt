# Calibrating Translation Decoding with Quality Estimation on LLMs

Official implementation based on the PyTorch and Hugging Face Transformers libraries.

[Human Annotation](https://huggingface.co/datasets/Calibration-Translation/Calibration-translation-human-eval)


**The full code, models, and human evaluation results are releasing ...**

# Installation
All experiments are tested with Python 3.8, torch 2.4.0

### Install Requirements
```
pip install -r requirements.txt
```

### Install Codebase
```
cd calibrating-llm-mt
pip install -U pip setuptools
pip install -e .
```

# Datasets
You can find datasets this paper involved here:

[calibration dataset](./src/llama_recipes/customer_data/calibration), [wmt24_testset](./src/llama_recipes/customer_data/wmt24_testset), [wmt22_metric_testset](./src/llama_recipes/customer_data/da_dataset)


# Quick Run
You can reproduce the results of applying calibration on TowerInstruct-Mistral-7B in Table-1. Training will 
takes around 1 GPU hour on H100. 
```
cd experiments
sh run.sh
```


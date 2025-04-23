# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import shutil
import sys
import time
import fire
import torch
from tqdm import tqdm

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.utils.dataset_utils import get_vllm_translation_dataset

from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    train_config as TRAIN_CONFIG,
)

from llama_recipes.utils.config_utils import get_dataloader_kwargs, update_config

from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create the directory
    os.makedirs(path)


def main(
    model_name,
    peft_model: str = None,
    merged_model_path: str = None,
    quantization: str = None,           # Options: 4bit, 8bit
    max_new_tokens=1000,                # The maximum numbers of tokens to generate
    seed: int = 42,                     # seed value for reproducibility
    do_sample: bool = True,             # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,             # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,           # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,                    # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,    # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,
    use_fast_kernels: bool = False,
    lang_pairs: str = None,
    output_dir: str = None,
    **kwargs,
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # Update the configuration for the training and sharding process
    test_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((test_config, fsdp_config), **kwargs)

    model = load_model(model_name, quantization, use_fast_kernels, **kwargs)

    if test_config.preload_peft_dir is not None:
        print("Load and merge pre_peft...")
        model = load_peft_model(model, test_config.preload_peft_dir)
        model = model.merge_and_unload()

    if peft_model:
        print("Load and merge peft...")
        model = load_peft_model(model, peft_model)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # merged_model_path = "/gpfs/work4/0/gus20642/dwu18/calibration/ALMA-1"
    merged_model_path = test_config.merged_model_path
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    # free model
    del model
    torch.cuda.empty_cache()

    # Load Model
    llm = LLM(model=merged_model_path, tensor_parallel_size=1)
    params = BeamSearchParams(beam_width=5, max_tokens=250)

    # Generate Output
    output = {}
    if lang_pairs is not None:
        lang_pairs = lang_pairs.split(',')
    else:
        lang_pairs = test_config.lang_pairs

    for lang_pair in lang_pairs:
        # Get test data
        print("Processing {} ...".format(lang_pair), flush=True)
        dataset_test = get_vllm_translation_dataset(test_config.dataset, split="test", lang_pairs=[lang_pair])
        print(f"--> Test Set Length = {len(dataset_test)}", flush=True)

        # Batch Inference
        start = time.perf_counter()
        best_translations = []
        cnt = 0
        for i in range(0, len(dataset_test), test_config.val_batch_size):
            prompts = dataset_test['prompt']
            batch_prompts = prompts[i:i + test_config.val_batch_size]
            results = llm.beam_search(batch_prompts, params)

            for beam_output, prompt in zip(results, batch_prompts):
                best_sequence = max(beam_output.sequences, key=lambda seq: seq.cum_logprob)  # Highest probability
                chinese_translation = best_sequence.text.split(prompt)[-1].strip("</s>").strip()
                best_translations.append(chinese_translation)

            cnt += 1
            if cnt % 10 == 0:
                print("process {} samples ...".format(cnt * test_config.val_batch_size))

        output[lang_pair] = best_translations

        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)

        # dump results
        src, tgt = lang_pair.split("-")
        create_clean_dir(os.path.join(output_dir, lang_pair))
        output_file = os.path.join(output_dir, lang_pair, "hyp.{}-{}.{}".format(src, tgt, tgt))
        with open(output_file, 'w') as fout:
            for line in output[lang_pair]:
                fout.write(line.strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)

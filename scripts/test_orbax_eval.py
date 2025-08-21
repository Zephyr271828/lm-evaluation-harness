# import hydra
# requirements: 
# pip install sacrebleu accelerate peft 
import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse

from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.orbax_lm import OrbaxLM

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations

from jax.sharding import Mesh
from jax.experimental import mesh_utils


PPL_TASKS = [
    # "c4",
    "wikitext",
    "wikitext2",
    # "cnn_dailymail",
    # "dclm"
]

TASK_CONFIG = {
    "winogrande": {
        # "num_fewshot": 5,
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    "arc_challenge": {
        # "num_fewshot": 25,
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    "hellaswag": {
        # "num_fewshot": 10,
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    "truthfulqa_mc1": {
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    "truthfulqa_mc2": {
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    "piqa": {
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    "sciq": {
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    "boolq": {
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    "arc_easy": {
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    "anli_r1": {
        "num_fewshot": 0,
        "acc_key": None,
    },
    "anli_r2": {
        "num_fewshot": 0,
        "acc_key": None,
    },
    "anli_r3": {
        "num_fewshot": 0,
        "acc_key": None,
    },
    "openbookqa": {
        "num_fewshot": 0,
        "acc_key": None,
    },
    "rte": {
        "num_fewshot": 0,
        "acc_key": None,
    },
    "mmlu": {
        # "num_fewshot": 5,
        "num_fewshot": 0,
        "acc_key": None,
    },
    "record": {
        "num_fewshot": 0,
        "acc_key": None,
    },
}

ACC_TASKS = TASK_CONFIG.keys()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ppl_enc(task, tokenizer):
    if task == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt')
    elif task == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt')
    elif task == 'cnn_dailymail':
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:16384][text_column]), return_tensors='pt')
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt')
    elif task == 'dclm':
        # data_paths = [
        #     '/datasets/dclm_baseline_1_0/dclm_baseline_1.0.val.jsonl',
        #     '/vast/yx3038/datasets/dclm/dclm_baseline_1.0_shuffled/dclm_baseline_1.0.val.jsonl'
        # ]   
        # for data_path in data_paths:
        #     if os.path.exists(data_path):
        #         dataset = load_dataset(
        #             "json",
        #             data_files={"train": data_path},
        #             split="train",
        #             verification_mode="no_checks"
        #         )
        #         text_column = "text"
        #         testenc = tokenizer.encode(" ".join(dataset[:1400][text_column]), return_tensors='pt')
        #         break
            

        dataset = load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            data_files="global-shard_05_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst",
            split="train",
            verification_mode="no_checks",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt')
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model, 
    tokenizer, 
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 2048
):
    # devices_in_data_fsdp = model.devices_in_data_fsdp
    # if batch_size % devices_in_data_fsdp != 0:
    #     print(f"🔁 Adjusting batch_size {batch_size} → {devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)} for device mesh compatibility.")
    #     batch_size = devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)
    
    ppl_res = {}
    for task in tasks:
        testenc = get_ppl_enc(task, tokenizer)
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = max_length
        nsamples = min(testenc.numel() // seq_len, calib_size)
        with torch.no_grad():
            for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
                j = min(i + bs, nsamples)
                inputs = testenc[:,(i * seq_len):(j * seq_len)]
                inputs = inputs.reshape(j - i, seq_len)
                # import pdb; pdb.set_trace()
                
                outputs = model.forward(inputs)
                if hasattr(outputs, "logits"):
                    lm_logits = outputs.logits
                else:
                    lm_logits = outputs
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss().to(shift_logits.device)
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                tot_loss += loss.item() * seq_len * (j - i)
                tot_tokens += seq_len * (j - i)
                
            ppl_res[task] = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
            print(task, ppl_res[task])
                
    return ppl_res

def get_acc(model, tokenizer, tasks):
    # lm_eval_model = models.orbax_lm.HFLM(
    #     pretrained=model, 
    #     tokenizer=tokenizer,
    #     generation_kwargs={
    #         "do_sample": True,
    #         "temperature": 0.2,
    #         "top_p": 0.95,
    #     }
    # )
    acc_res = {}
    for task in tasks:
        res = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=TASK_CONFIG[task]["num_fewshot"],
            max_batch_size=32,
            log_samples=True,
            # task_kwargs={"limit": 256}, 
            confirm_run_unsafe_code=True,
        )
        
        print(res['results'][task])
        acc_key = TASK_CONFIG[task]["acc_key"]
        if acc_key is not None:
            acc_res[task] = res['results'][task][acc_key]

    return acc_res

def cast_orbax_state_to_bf16(orbax_state):
    casted_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x,
        orbax_state.params
    )
    orbax_state = orbax_state.replace(params=casted_params)
    return orbax_state

def main(config, test_args):
    tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)
    
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    orbax_model = models.Transformer(config, mesh, quant=quant)
    orbax_state, _ = maxtext_utils.setup_decode_state(orbax_model, config, rng1, mesh, None)
    
    orbax_state = cast_orbax_state_to_bf16(orbax_state)
    
    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
        orbax_model, None, config, rng1, mesh, is_training=False
    )

    model = OrbaxLM(orbax_model, orbax_state, tokenizer, config, state_mesh_shardings, mesh)
    
    # ppl_res = get_ppl(
    #     model, 
    #     tokenizer, 
    #     # batch_size=config.global_batch_size_to_train_on, 
    #     batch_size=1,
    #     max_length=config.max_target_length, 
    #     tasks=PPL_TASKS
    # )
    # ppl_res = get_ppl(model, tokenizer, tasks=['dclm'])
    # print(ppl_res)

    acc_res = get_acc(model, tokenizer, tasks=TASK_CONFIG.keys())
    # acc_res = get_acc(model, tokenizer, tasks=['winogrande'])
    print(acc_res)
    
if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, required=False, default=0.1)
    parser.add_argument("--rtol", type=float, required=False, default=0.1)
    parser.add_argument("--token_size", type=int, required=False)
    parser.add_argument("--max_kl_div", type=float, required=False, default=None)
    parser.add_argument("--golden_logits_path", type=str, required=False, default="")
    parser.add_argument("--hf_model_path", type=str, required=False, default="")
    parser.add_argument("--run_hf_model", type=bool, required=False, default=False)
    test_args, _ = parser.parse_known_args()

    # Remove args defined in this test file to avoid error from pyconfig
    model_args = sys.argv
    to_remove_args = [
        "--atol",
        "--rtol",
        "--token_size",
        "--max_kl_div",
        "--golden_logits_path",
        "--hf_model_path",
        "--run_hf_model",
    ]
    for arg in to_remove_args:
        model_args = [s for s in model_args if not s.startswith(arg)]

    cfg = pyconfig.initialize(model_args)
    main(cfg, test_args)
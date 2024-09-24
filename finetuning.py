#!/usr/bin/env python
# coding: utf-8

# Simple benchmarking script that does fine-tuning on a given Hugging
# Face model with IMDB movie reviews
#
# Adapted from the exercises of the LUMI AI workshop course:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop

import argparse
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
from peft import get_peft_model, LoraConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help="The pre-trained model from Hugging Face to use as basis: "
        "https://huggingface.co/models"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="The number of CPU worker processes to use.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="If set, continue from a previously interrupted run. "
        "Otherwise, overwrite existing checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="The number of training steps.",
    )
    parser.add_argument(
        "--peft",
        action='store_true',
        help="Use PEFT: https://huggingface.co/docs/peft/index"
    )
    parser.add_argument(
        "--4bit",
        dest="bnb_4bit",
        action='store_true',
        help="Use 4bit quantization with bitsandbytes: "
        "https://huggingface.co/docs/bitsandbytes/main/en/index"
    )
    args, _ = parser.parse_known_args()

    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Then we determine the device on which to train the model.
    if rank == 0:
        print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        print(f"Using GPU {local_rank}, device name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"No GPU found, using CPU instead. (Rank: {local_rank})")
        device = torch.device("cpu")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} has to be a multiple of "
              f"the number of GPUs={world_size}!")
        sys.exit(1)

    # We also ensure that output paths exist
    model_name = args.model.replace('/', '_')

    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, model_name)

    # Load the IMDb data set
    train_dataset = load_dataset(
        "imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True
    )
    eval_dataset = load_dataset(
        "imdb", split="test", trust_remote_code=False, keep_in_memory=True
    )

    # #### Loading the model
    # Let's start with getting the appropriate tokenizer.
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    # Load the actual base model from Hugging Face
    if rank == 0:
        print("Loading model and tokenizer")

    quantization_config = None
    if args.bnb_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        quantization_config = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device
        )

    if args.peft:
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
        #     lora_dropout=0.1
        # )

        # LoRA config from here:
        # https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_fsdp_qlora.py#L128
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
            )
        model = get_peft_model(model, peft_config)
        print("Using PEFT")
        model.print_trainable_parameters()

    #model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.resume,
        # save_strategy="no",  # good for testing
        save_strategy="steps",   # use these if you actually want to save the model
        save_steps=400,
        save_total_limit=4,
        # eval_strategy="steps",
        # eval_steps=200,  # compute validation loss every 200 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=train_batch_size // world_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        # report_to=["tensorboard"],  # log statistics for tensorboard
        ddp_find_unused_parameters=False,
    )

    # #### Setting up preprocessing of training data

    # IMDb examples are presented as a dictionary:
    # {
    #    'text': the review text as a string,
    #    'label': a sentiment label as an integer,
    # }.
    # We tokenize the text and add the special token for indicating the end of the
    # text at the end of each review. We also truncate reviews to a maximum
    # length to avoid excessively long sequences during training.
    # As we have no use for the label, we discard it.
    max_length = 256

    def tokenize(x):
        texts = [example + tokenizer.eos_token for example in x["text"]]
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )

    train_dataset_tok = train_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    eval_dataset_tok = eval_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        num_proc=training_args.dataloader_num_workers,
    )

    # We split a small amount of training data as "validation" test
    # set to keep track of evaluation of the loss on non-training data
    # during training.  This is purely because computing the loss on
    # the full evaluation dataset takes much longer.
    train_validate_splits = train_dataset_tok.train_test_split(
        test_size=1000, seed=42, keep_in_memory=True
    )
    train_dataset_tok = train_validate_splits["train"]
    validate_dataset_tok = train_validate_splits["test"]

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tok,
        eval_dataset=validate_dataset_tok,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if rank == 0:
        print()
        print("Training done, you can find all the model checkpoints in", output_dir)

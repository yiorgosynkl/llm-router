import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
import argparse
import gradio as gr

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_probs(text, model, tokenizer, max_length=512):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = encoding["attention_mask"].to(device)
    ids = encoding["input_ids"].to(device)
    logits = model(ids, token_type_ids=None, attention_mask=mask)[0]
    probs = torch.nn.functional.softmax(
        logits, dim=1
    )  # softmax to convert logits to probabilities
    return probs.flatten().tolist()


def infer_single(text, model, tokenizer, id2label):
    probs = get_probs(text, model=model, tokenizer=tokenizer)
    res = {label: probs[i] for i, label in id2label.items()}
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="../models/bert-base-uncased-router-finetuning-20240722T133228-save",
        help="The directory where the BERT model weights are stored (used `model.save_pretrained(...)`",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="bert-base-uncased",
        help="The tokenizer name in hugging face",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    save_directory = os.path.join(os.getcwd(), args.model_save_dir)
    model = BertForSequenceClassification.from_pretrained(save_directory)
    model.to(device)

    def demo_infer_single(text):
        return infer_single(
            text,
            model=model,
            tokenizer=tokenizer,
            id2label={0: "ROUTE_TO_INFERIOR", 1: "ROUTE_TO_SUPERIOR"},
        )

    demo = gr.Interface(fn=demo_infer_single, inputs="text", outputs="label")
    demo.launch()

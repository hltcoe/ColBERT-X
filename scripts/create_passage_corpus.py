#!/usr/bin/env python
import os
import re
import json
import argparse
from tqdm import tqdm
from typing import Dict
from transformers import AutoTokenizer
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) 

def strip_newlines(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text

def process_documents(args) -> Dict[str, str]:
    num_docs = sum(1 for line in open(args.corpus))
    with open(args.corpus) as f:
        for line in tqdm(f, total=num_docs):
            temp = json.loads(line)
            doc_id = temp[args.docid]
            title = strip_newlines(temp[args.title]) 
            text = strip_newlines(temp[args.body])
            doc_text = title + " " + text if title else text
            if args.lower:
                doc_text = doc_text.lower()
            encoded_tokens = args.tokenizer.encode(doc_text)[1:-1]
            s, e, idx = 0, 0, 0
            while s < len(encoded_tokens):
                e = s + args.length
                if e >= len(encoded_tokens):
                    e = len(encoded_tokens)
                p = args.tokenizer.decode(encoded_tokens[s:e])
                pass_id = f"{doc_id}_{idx}"
                s = s + args.length - args.stride
                yield p, pass_id
                idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that generates overlapping passages from a JSONL document collection")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--length", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--docid", type=str, default="id")
    parser.add_argument("--title", type=str, default="title")
    parser.add_argument("--body", type=str, default="text")
    parser.add_argument("--lower", action="store_true", default=False) 
    args = parser.parse_args()

    args.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    pass_file = os.path.join(args.root, "collection_passages.tsv")
    map_file = os.path.join(args.root, "mapping.tsv")
    
    with open(pass_file, "w") as f, open(map_file, "w") as g:
        for idx,(line1,line2) in enumerate(process_documents(args)):
            f.write(f"{idx}\t{line1}\n")
            g.write(f"{idx}\t{line2}\n")

import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from xlmr_colbert.parameters import DEVICE
from xlmr_colbert.modeling.colbert import ColBERT
from xlmr_colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
    colbert = ColBERT.from_pretrained('xlm-roberta-large',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    
    colbert.roberta.resize_token_embeddings(len(colbert.tokenizer))

    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint

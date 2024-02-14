import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.infra import ColBERTConfig, Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints, find_last_checkpoint, load_checkpoint_misc

from functools import partial
from colbert.modeling.tokenization.utils import legacy_tensorize_triples

def train(config: ColBERTConfig, triples, queries=None, collection=None):
    # lagacy training code is only compatible with
    assert config.nway == 2

    if config.resume:
        config.checkpoint = config.checkpoint or find_last_checkpoint(config.checkpoint_path_)
    else: 
        config.checkpoint = config.checkpoint or config.model_name



    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    
    torch.cuda.manual_seed_all(12345)

    
    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    assert config.accumsteps == 1
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    
    reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    # reader.tensorize_triples = partial(legacy_tensorize_triples, reader.query_tokenizer, reader.doc_tokenizer)

    # if config.rank not in [-1, 0]:
    #     torch.distributed.barrier()

    colbert = ColBERT(name=config.checkpoint, colbert_config=config)

    
    # TODO: add load optimizer back     
    # if config.checkpoint is not None:
    #     # assert config.resume_optimizer is False, "TODO: This would mean reload optimizer too."
    #     if not config.resume_optimizer:
    #         print_message(f"#> Starting from checkpoint {config.checkpoint} -- but NOT the optimizer!")
    #     else:
    #         print_message(f"#> Starting from checkpoint {config.checkpoint}")

    #     checkpoint = torch.load(config.checkpoint, map_location='cpu')

    #     try:
    #         colbert.load_state_dict(checkpoint['model_state_dict'])
    #     except:
    #         print_message("[WARNING] Loading checkpoint with strict=False")
    #         colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # if config.rank == 0:
    #     torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()


    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    # Attempt to load optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    # if config.resume_optimizer:
    #     assert config.checkpoint is not None
    #     print_message("#> Loading the optimizer")
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.zero_grad()

    amp = MixedPrecisionManager(config.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages, _ in BatchSteps:
            with amp.context():
                # queries = ( torch.concat((queries[0], queries[0])), torch.concat((queries[1], queries[1])) )
                # passages = ( torch.concat((passages[0][0::2], passages[0][1::2])), torch.concat((passages[1][0::2], passages[1][1::2])) )

                # queries = (
                #     queries[0].repeat_interleave(2, dim=0).contiguous(),
                #     queries[1].repeat_interleave(2, dim=0).contiguous()
                # )

                scores = colbert(queries, passages)
                # print(scores.tolist())
                
                scores = scores.view(-1, config.nway)

                # .view(-1, config.nway)
                
                # .view(2, -1).permute(1, 0)

                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if config.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            # num_examples_seen = (batch_idx - start_batch_idx) * config.bsize * config.nranks
            # elapsed = float(time.time() - start_time)

            # log_to_mlflow = (batch_idx % 20 == 0)
            # Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            # Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            # Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            # Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            print_message(batch_idx, avg_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1)
            
    manage_checkpoints(config, colbert, optimizer, batch_idx+1, consumed_all_triples=True)
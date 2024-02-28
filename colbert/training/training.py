import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher, MultiLangBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints, find_last_checkpoint, load_checkpoint_misc



def train(config: ColBERTConfig, triples, queries=None, collection=None):
    if config.resume:
        config.checkpoint = config.checkpoint or find_last_checkpoint(config.checkpoint_path_)
    else: 
        config.checkpoint = config.checkpoint or config.model_name

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if hasattr(collection, 'load_all_collections'):
            collection.load_all_collections()
            
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        elif config.multilang:
            reader = MultiLangBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        reader = LazyBatcher(config, triples, None, None, (0 if config.rank == -1 else config.rank), config.nranks)
        # raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    if config.resume or config.resume_optimizer:
        start_batch_idx, optimizer_state_dict = load_checkpoint_misc(config)
        optimizer.load_state_dict(optimizer_state_dict)

        if config.resume:
            reader.skip_to_batch(start_batch_idx)

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                if len(batch) == 3:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                else:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores: torch.Tensor = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = F.log_softmax(target_scores, dim=-1)

                    log_scores = F.log_softmax(scores, dim=-1)
                    
                    if config.kd_loss == 'KLD':
                        loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                    elif config.kd_loss == 'MSE':
                        loss = nn.MSELoss()(log_scores, target_scores)
                    else:
                        raise ValueError(f"Unsupported kd loss function: {config.kd_loss}")
                else:
                    # if config.rank < 1:
                    #     print(batch)
                    #     print(scores, labels[:scores.size(0)])
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives: # Eugene: pretty much broken...
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss
                
                if config.multilang and not config.nolangreg:
                    ## all passages might be too much...
                    # lang_scores: torch.Tensor = scores.view(-1, config.bsize // config.accumsteps, config.nway).permute(1, 2, 0).flatten(start_dim=0, end_dim=1)
                    lang_scores: torch.Tensor = log_scores.view(-1, config.bsize // config.accumsteps, config.nway).permute(1, 2, 0).flatten(start_dim=0, end_dim=1)
                    
                    # try only the most relevant passage
                    # most_rel = target_scores.argmax(dim=-1)
                    # lang_scores = scores.index_select(-1, most_rel).diag().view(reader.nlang, -1).T
                    # print(scores)
                    # print(lang_scores)

                    # loss += (lang_scores.max(dim=-1).values - lang_scores.min(dim=-1).values).mean()
                    # lang_loss = (lang_scores - lang_scores.mean(dim=-1).unsqueeze(1)).norm(dim=-1).mean() 
                    # scale = min(1 / (200*(loss.item()**2)), 1.)
                    
                    lang_loss = []
                    for i in range(reader.nlang):
                        for j in range(i, reader.nlang):
                            lang_loss.append(symmetric_divergence(lang_scores[:, i], lang_scores[:, j]))
                    lang_loss = torch.stack(lang_loss).mean()
                    
                    if config.rank < 1:
                        print(f"#>>> loss={loss.item()}+{lang_loss.item()}")
                        
                    loss += lang_loss


                loss = loss / config.accumsteps

            if config.rank < 1:
                if len(target_scores) and not config.ignore_scores:
                    highest_avg, lowest_avg = scores.max(dim=-1).values.mean().item(), scores.min(dim=-1).values.mean().item()
                    print(f"#>>>   {highest_avg:.2f}, {lowest_avg:.2f} \t\t|\t\t {highest_avg-lowest_avg:.2f}")
                else:
                    print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


def symmetric_divergence(dist_a, dist_b, alpha=0.5, log_target=True):
    # Jensen–Shannon divergence
    return alpha * nn.KLDivLoss(reduction='batchmean', log_target=log_target)(dist_a, dist_b) + \
           (1-alpha) * nn.KLDivLoss(reduction='batchmean', log_target=log_target)(dist_b, dist_a)


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)

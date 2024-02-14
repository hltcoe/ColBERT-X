import os
import ujson
from random import shuffle
from tqdm.auto import tqdm

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

from colbert.utils.runs import Run

from scipy.stats import beta
import numpy as np
def sampling_on_beta(items, nway: int, step: int, max_step: int, max_beta: float):
    samp_dist: np.ndarray = beta(1.0, 1.+step/max_step*(max_beta-1.)).pdf(np.arange(len(items))/len(items))
    samp_dist = samp_dist / samp_dist.sum()
    idx = np.random.choice(len(items), size=nway, replace=False, p=samp_dist)
    return [ items[i] for i in idx ]


class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries=None, collection=None, rank=0, nranks=1):
        self.config = config

        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway
        self.shuffle_pasages = config.shuffle_passages
        self.sampling_max_beta = config.sampling_max_beta
        self.over_one_epoch = config.over_one_epoch

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        if self.shuffle_pasages:
            self.triples = Examples.cast(triples, nway=100000).tolist(rank, nranks)
        else:
            self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)

        # print(len(self.triples), self.triples[:10])
        self.queries = queries and Queries.cast(queries)
        self.collection = collection and Collection.cast(collection)

        self.skipping = False # only used when skipping batches

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        # TODO: Eugene [distillation] do some curriculum for this sampling
        # Curriculum Learning for Dense Retrieval Distillation: Zeng, Zamani, Vinay
        query, *passages = self.triples[ idx % len(self.triples) ]
        if self.shuffle_pasages:
            if self.sampling_max_beta > 1:
                return [ 
                    query, 
                    *sampling_on_beta(
                        passages, self.nway, idx//self.bsize, 
                        self.config.maxsteps, self.sampling_max_beta
                    ) 
                ]
            else:
                shuffle(passages)
        return [ query, *passages ]

    def __next__(self):
        # offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        offset, endpos = self.position, self.position + self.bsize
        self.position = endpos

        if not self.over_one_epoch and offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self[position]
            if self.skipping:
                continue

            pids = pids[:self.nway]

            query = self.queries[query] if self.queries else query

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = self.collection[pids] if self.collection else pids

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
        
        if self.skipping:
            return 

        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        # print(all_queries)

        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):        
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)

    def skip_to_batch(self, batch_idx):
        print(f'>>> Skipping to batch #{batch_idx} for training.')
        self.skipping = True
        for _ in tqdm(range(batch_idx), desc='skipping', dynamic_ncols=True):
            next(self)
        self.skipping = False # !!! important to turn it back on
        print(f">>> Done skipping.")

class MultiLangBatcher(LazyBatcher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.collection is not None
        
        self.nlang = len(next(iter(self.collection)))

    def collate(self, queries, passages, scores):
        # each passage has self.nlang variants --> flatten them 
        assert all( len(p) == self.nlang for p in passages )
        
        queries = queries*self.nlang
        passages = sum(map(list, zip(*passages)), [])

        if len(scores) != 0:
            scores = scores*self.nlang

        assert len(queries) == self.bsize * self.nlang
        assert len(passages) == self.nway * self.bsize * self.nlang

        # print(len(queries), len(passages), len(scores))
        # print(queries, passages, scores)
        # print(passages)

        return self.tensorize_triples(queries, passages, scores, self.bsize * self.nlang // self.accumsteps, self.nway)
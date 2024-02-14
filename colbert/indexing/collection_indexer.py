import os
import tempfile
from typing import Dict, Iterable
import tqdm
import time
import ujson
import torch
import pickle
import random
from collections import defaultdict
import more_itertools
from pathlib import Path

try:
    import faiss
except ImportError as e:
    print("WARNING: faiss must be imported for indexing")

import numpy as np
import torch.multiprocessing as mp
from colbert.infra.config.config import ColBERTConfig

import colbert.utils.distributed as distributed

from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats
from colbert.modeling.checkpoint import Checkpoint
from colbert.data.collection import Collection

from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.loaders import load_doclens
from colbert.indexing.utils import optimize_ivf
from colbert.utils.utils import flatten, print_message

from colbert.indexing.codecs.residual import ResidualCodec

# Eugene: deprecated
# def encode(config, collection, shared_lists, shared_queues):
#     encoder = CollectionIndexer(config=config, collection=collection)
#     encoder.run(shared_lists)

def reuse_prepare(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection, no_checkpoint=True)
    encoder.steps(shared_lists, -1)

def sample(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection)
    encoder.steps(shared_lists, 0)

def kmeans(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection, no_checkpoint=True)
    encoder.steps(shared_lists, 1)

def encode(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection)
    encoder.steps(shared_lists, 2)

def finalize(config, collection, shared_lists, shared_queues):
    encoder = CollectionIndexer(config=config, collection=collection)
    encoder.steps(shared_lists, 3)


class CollectionIndexer():
    '''
    Given a collection and config, encode collection into index and
    stores the index on the disk in chunks.
    '''
    def __init__(self, config: ColBERTConfig, collection, no_checkpoint=False):
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = self.config.total_visible_gpus > 0

        if self.config.rank == 0:
            self.config.help()

        self.collection = Collection.cast(collection)

        if not no_checkpoint:
            self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config)
            if self.use_gpu:
                self.checkpoint = self.checkpoint.cuda()

            self.encoder = CollectionEncoder(config, self.checkpoint)
        
        self.saver = IndexSaver(config)

        print_memory_stats(f'RANK:{self.rank}')

    def run(self, shared_lists):
        # FIXME Eugene: 
        # For some reason running `self.train` on rank=0 is either not sufficient to block other
        # child processes to use GPU or preventing FAISS from allocating memory on GPUs previously used by other child processes.
        # So to get around this, the pipeline needs to be separated into three parts -- setup(multiprocess), 
        # train(initiating from the main process), and the rest(multiprocessing). 
        # This would be controlled by the indexer. 
        # This function would be deprecated.

        with torch.inference_mode():
            self.setup() # Computes and saves plan for whole collection
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            if not self.config.resume or not self.saver.try_load_codec():
                self.train(shared_lists) # Trains centroids from selected passages
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            self.index() # Encodes and saves all tokens into residuals
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

            self.finalize() # Builds metadata and centroid to passage mapping
            distributed.barrier(self.rank)
            print_memory_stats(f'RANK:{self.rank}')

    def steps(self, shared_lists, step: int):
        with torch.inference_mode():
            if step == -1:
                # meant to be in the main process
                self.setup_reuse()
            elif step == 0:
                self.setup()
                distributed.barrier(self.rank)
                print_memory_stats(f'RANK:{self.rank}')
            elif step == 1:
                # should use the main process for entering this block
                if not self.saver.try_load_codec():
                    assert self._try_load_plan()
                    self.train(shared_lists)
            elif step == 2:
                assert self._try_load_plan()
                assert self.saver.try_load_codec()
                self.index()
                distributed.barrier(self.rank)
                print_memory_stats(f'RANK:{self.rank}')
            
            elif step == 3:
                # should use the main process for entering this block
                assert self._try_load_plan()

                self.finalize()
                
                print_memory_stats(f'RANK:{self.rank}')
            else:
                Run().print_main(f"Step {step} does not exist -- ignoring")

    def setup(self):
        '''
        Calculates and saves plan.json for the whole collection.
        
        plan.json { config, num_chunks, num_partitions, num_embeddings_est, avg_doclen_est}
        num_partitions is the number of centroids to be generated.
        '''
        if self.config.resume:
            if self._try_load_plan():
                Run().print_main(f"#> Loaded plan from {self.plan_path}:")
                Run().print_main(f"#> num_chunks = {self.num_chunks}")
                Run().print_main(f"#> num_partitions = {self.num_partitions}")
                Run().print_main(f"#> num_embeddings_est = {self.num_embeddings_est}")
                Run().print_main(f"#> avg_doclen_est = {self.avg_doclen_est}")
                return

        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))

        # Saves sampled passages and embeddings for training k-means centroids later 
        sampled_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        # Select the number of partitions
        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * avg_doclen_est

        self.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))
        if self.config.max_num_partitions > 0:
            self.num_partitions = min(self.num_partitions, self.config.max_num_partitions)

        Run().print_main(f'Creaing {self.num_partitions:,} partitions.')
        Run().print_main(f'*Estimated* {int(self.num_embeddings_est):,} embeddings.')

        self._save_plan()

    def setup_reuse(self):
        if not self._try_load_plan(self.config.reuse_centroids_from):
            raise FileNotFoundError(f"Cannot find plans in index `{self.config.reuse_centroids_from}`")
        
        Run().print_main(f"#> Loaded existing plan from {self.config.reuse_centroids_from}:")
        Run().print_main(f"#> num_partitions = {self.num_partitions}")
        Run().print_main(f"#> avg_doclen_est = {self.avg_doclen_est} <-- not accurate but will not recalculate")
        Run().print_main(f"#> replacing other information with the current collection info")

        # replace collection information here
        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))

        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * self.avg_doclen_est
        self._save_plan()

        # and soft link the rest
        for fn in ['centroids', 'avg_residual', 'buckets']:
            os.symlink(os.path.join(self.config.reuse_centroids_from, f"{fn}.pt"), os.path.join(self.config.index_path_, f"{fn}.pt"))


    def _sample_pids(self):
        num_passages = len(self.collection)

        # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        # Keep in mind that, say, 15% still means at least 100k.
        # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
        # Then we subsample the vectors to 100 * num_partitions

        typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
        sampled_pids = 16 * np.sqrt(typical_doclen * num_passages)
        # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
        sampled_pids = min(1 + int(sampled_pids), num_passages)
        if self.config.max_sampled_pid > 0:
            sampled_pids = min(sampled_pids, self.config.max_sampled_pid)

        sampled_pids = random.sample(range(num_passages), sampled_pids)
        Run().print_main(f"# of sampled PIDs = {len(sampled_pids)} \t sampled_pids[:3] = {sampled_pids[:3]}")

        return list(sorted(set(sampled_pids)))

    def _sample_embeddings(self, sampled_pids):
        # local_pids = self.collection.enumerate(rank=self.rank)
        # local_sample = [passage for pid, passage in local_pids if pid in sampled_pids]

        # Note from Eugene: how sample is distributed across GPUs shouldn't matter -- as long as each example has a home is fine
        # local_sample = [passage for pid, passage in self.collection.enumerate(rank=self.rank) if pid in sampled_pids]

        # This uses random access from the Collection class -- should be much faster than iterating through the collection
        local_sample = [ self.collection[pid] for pid in sampled_pids[self.rank::self.nranks] ]

        local_sample_embs, doclens = self.encoder.encode_passages(local_sample)

        if torch.cuda.is_available():
            if torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
                torch.distributed.all_reduce(self.num_sample_embs)

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
                torch.distributed.all_reduce(avg_doclen_est)

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()
                torch.distributed.all_reduce(nonzero_ranks)
            else:
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()
        else:
            if torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()
                torch.distributed.all_reduce(self.num_sample_embs)

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()
                torch.distributed.all_reduce(avg_doclen_est)

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cpu()
                torch.distributed.all_reduce(nonzero_ranks)
            else:
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()

                avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()

                nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cpu()

        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        self.avg_doclen_est = avg_doclen_est

        Run().print(f'avg_doclen_est = {avg_doclen_est} \t len(local_sample) = {len(local_sample):,}')

        torch.save(local_sample_embs.half(), os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))

        return avg_doclen_est

    def _try_load_plan(self, path=None):
        config = self.config
        self.plan_path = os.path.join(config.index_path_, 'plan.json')
        path = os.path.join(path, 'plan.json') if path is not None else self.plan_path

        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    plan = ujson.load(f)
                except Exception as e:
                    return False
                if not ('num_chunks' in plan and
                        'num_partitions' in plan and
                        'num_embeddings_est' in plan and
                        'avg_doclen_est' in plan):
                    return False

                # TODO: Verify config matches
                self.num_chunks = plan['num_chunks']
                self.num_partitions = plan['num_partitions']
                self.num_embeddings_est = plan['num_embeddings_est']
                self.avg_doclen_est = plan['avg_doclen_est']

                if 'num_sample_embs' in plan:
                    self.num_sample_embs = plan['num_sample_embs']

            return True
        else:
            return False

    def _save_plan(self):
        if self.rank < 1:
            config = self.config
            self.plan_path = os.path.join(config.index_path_, 'plan.json')
            Run().print("#> Saving the indexing plan to", self.plan_path, "..")

            with open(self.plan_path, 'w') as f:
                d = {'config': config.export()}
                d['num_chunks'] = self.num_chunks
                d['num_partitions'] = self.num_partitions
                d['num_embeddings_est'] = self.num_embeddings_est
                d['avg_doclen_est'] = self.avg_doclen_est

                if hasattr(self, 'num_sample_embs'):
                    try:
                        d['num_sample_embs'] = int(self.num_sample_embs.detach().cpu())
                    except:
                        d['num_sample_embs'] = int(self.num_sample_embs)

                f.write(ujson.dumps(d, indent=4) + '\n')


    def train(self, shared_lists):
        if self.rank > 0:
            return

        sample, heldout = self._concatenate_and_split_sample()

        centroids = self._train_kmeans(sample, shared_lists)

        print_memory_stats(f'RANK:{self.rank}')
        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)

        print_message(f'avg_residual = {avg_residual}')

        # Compute and save codec into avg_residual.pt, buckets.pt and centroids.pt
        codec = ResidualCodec(config=self.config, centroids=centroids, avg_residual=avg_residual,
                              bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
        self.saver.save_codec(codec)

    def _concatenate_and_split_sample(self):
        print_memory_stats(f'***1*** \t RANK:{self.rank}')

        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        # if not hasattr(self, 'num_sample_embs'):
        #     self.num_sample_embs = sum([ 
        #         torch.load( os.path.join(self.config.index_path_, f'sample.{r}.pt') ).size(0) for r in range(self.config.nranks) 
        #     ])

        # forced it to float32 when storing already
        sample = torch.empty(self.num_sample_embs, self.config.dim, dtype=torch.float32)
        
        # load with shuffle to avoid materialize them twice
        shuffle_order = torch.randperm(sample.size(0))

        offset = 0
        print_message(f'Loading {self.nranks} sample embedding files, expecting {self.num_sample_embs} samples')
        for r in range(self.nranks):
            sub_sample_path = os.path.join(self.config.index_path_, f'sample.{r}.pt')
            sub_sample: torch.Tensor = torch.load(sub_sample_path)

            # os.remove(sub_sample_path)

            endpos = offset + sub_sample.size(0)
            # sample[offset:endpos] = sub_sample
            sample[ shuffle_order[offset:endpos] ] = sub_sample.float()
            offset = endpos

        assert endpos == sample.size(0), (endpos, sample.size())

        print_memory_stats(f'***2*** \t RANK:{self.rank}')

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        # sample = sample[torch.randperm(sample.size(0))]

        print_memory_stats(f'***3*** \t RANK:{self.rank}')

        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, sample_heldout = sample.split([sample.size(0) - heldout_size, heldout_size], dim=0)

        print_memory_stats(f'***4*** \t RANK:{self.rank}')

        return sample, sample_heldout

    def _train_kmeans(self, sample, shared_lists):
        if self.use_gpu:
            torch.cuda.empty_cache()

        do_fork_for_faiss = False  # set to True to free faiss GPU-0 memory at the cost of one more copy of `sample`.

        args_ = [self.config.dim, self.num_partitions, self.config.kmeans_niters]

        if do_fork_for_faiss:
            # For this to work reliably, write the sample to disk. Pickle may not handle >4GB of data.
            # Delete the sample file after work is done.

            # shared_lists[0][0] = sample
            shared_lists[0].append( sample )
            return_value_queue = mp.Queue()

            args_ = args_ + [shared_lists, return_value_queue]
            proc = mp.Process(target=compute_faiss_kmeans, args=args_)

            proc.start()
            centroids = return_value_queue.get()
            proc.join()

        else:
            args_ = args_ + [[[sample]]]
            centroids = compute_faiss_kmeans(*args_)

        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        if self.use_gpu:
            centroids = centroids.half()
        else:
            centroids = centroids.float()

        return centroids

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodec(config=self.config, centroids=centroids, avg_residual=None)

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device='cuda' if self.use_gpu else 'cpu')
        heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct, out_device='cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            heldout_avg_residual = heldout.cuda() - heldout_reconstruct
        else:
            heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        print([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2 ** self.config.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        print_message(
            f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}")
        print_message(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

        # EVENTAULLY: Compare the above with non-heldout sample. If too different, we can do better!
        # sample = sample[subsample_idxs]
        # sample_reconstruct = get_centroids_for(centroids, sample)
        # sample_avg_residual = (sample - sample_reconstruct).mean(dim=0)

    def index(self):
        '''
        Encode embeddings for all passages in collection.
        Each embedding is converted to code (centroid id) and residual.
        Embeddings stored according to passage order in contiguous chunks of memory.

        Saved data files described below:
            {CHUNK#}.codes.pt:      centroid id for each embedding in chunk
            {CHUNK#}.residuals.pt:  16-bits residual for each embedding in chunk
            doclens.{CHUNK#}.pt:    number of embeddings within each passage in chunk
        '''
        with self.saver.thread():
            batches = self.collection.enumerate_batches(rank=self.rank)
            for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=self.rank > 0):
                if self.config.resume and self.saver.check_chunk_exists(chunk_idx):
                    Run().print_main(f"#> Found chunk {chunk_idx} in the index already, skipping encoding...")
                    continue
                # Encode passages into embeddings with the checkpoint model
                embs, doclens = self.encoder.encode_passages(passages) 
                if self.use_gpu:
                    assert embs.dtype == torch.float16
                else:
                    assert embs.dtype == torch.float32
                    embs = embs.half()

                Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                                 f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")

                self.saver.save_chunk(chunk_idx, offset, embs, doclens) # offset = first passage index in chunk
                del embs, doclens

    def finalize(self):
        '''
        Aggregates and stores metadata for each chunk and the whole index
        Builds and saves inverse mapping from centroids to passage IDs

        Saved data files described below:
            {CHUNK#}.metadata.json: [ passage_offset, num_passages, num_embeddings, embedding_offset ]
            metadata.json: [ num_chunks, num_partitions, num_embeddings, avg_doclen ]
            inv.pid.pt: [ ivf, ivf_lengths ]
                ivf is an array of passage IDs for centroids 0, 1, ...
                ivf_length contains the number of passage IDs for each centroid
        '''
        if self.rank > 0:
            return

        self._check_all_files_are_saved()
        self._collect_embedding_id_offset()

        if not os.path.exists(os.path.join(self.config.index_path_, 'ivf.pid.pt')):
            self._build_ivf()
        else:
            Run().print_main("Found ivf file -- skip building")
        self._update_metadata()

    def _check_all_files_are_saved(self):
        Run().print_main(f"#> Checking all files were saved, expecting {self.num_chunks}...")
        success = True
        for chunk_idx in range(self.num_chunks):
            if not self.saver.check_chunk_exists(chunk_idx):
                success = False
                Run().print_main(f"#> ERROR: Could not find chunk {chunk_idx}!")
                #TODO: Fail here?
        if success:
            Run().print_main("Found all files!")

    def _collect_embedding_id_offset(self):
        passage_offset = 0
        embedding_offset = 0

        self.embedding_offsets = []

        for chunk_idx in range(self.num_chunks):
            metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
            Run().print_main(f"#> Read file: `{metadata_path}`")

            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                chunk_metadata['embedding_offset'] = embedding_offset
                self.embedding_offsets.append(embedding_offset)

                assert chunk_metadata['passage_offset'] == passage_offset, (chunk_idx, passage_offset, chunk_metadata)

                passage_offset += chunk_metadata['num_passages']
                embedding_offset += chunk_metadata['num_embeddings']

            with open(metadata_path, 'w') as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + '\n')

        self.num_embeddings = embedding_offset
        assert len(self.embedding_offsets) == self.num_chunks

    def _build_ivf(self):
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.

        if self.config.use_lagacy_build_ivf: 
            return self._legacy_build_ivf()

        Run().print_main("#> Building IVF...")

        #codes = torch.zeros(self.num_embeddings,).long()
        #print_memory_stats(f'RANK:{self.rank}')

        all_doclens = load_doclens(self.config.index_path_, flatten=False)
        pid_offsets = torch.cumsum(torch.tensor([0] + [ 
            len(doc) for doc in all_doclens 
        ]), dim=0 )[:-1]

        from multiprocessing import Pool
        from functools import partial
        
        working_dir = tempfile.mkdtemp() if self.config.ivf_use_tempdir else self.config.index_path_

        work_ = partial(build_local_ivf, self.config.index_path_, working_dir, self.num_partitions)
        
        Run().print_main(f"#> Building local IVFs with {self.config.ivf_num_processes} processes...")

        with Pool(self.config.ivf_num_processes) as pool:
            list(pool.map(work_, [
                (pid_offsets[chunk_idx], all_doclens[chunk_idx], chunk_idx)
                for chunk_idx in range(self.num_chunks)
            ]))
        
        # local_cluster2pids: Iterable[Dict[int, set]] = peekable(map(
        #     lambda chunk_idx: torch.load(os.path.join(working_dir, f"localivf.{chunk_idx}.pt")),
        #     range(self.num_chunks)
        # ))
        
        # Run().print_main(f"#> Combining local IVFs...")
        # cluster2pid = [ [] for _ in range(self.num_partitions) ]
        # for local in tqdm.tqdm(local_cluster2pids, total=self.num_chunks):
        #     # for i in range(self.num_partitions):
        #     for idx, pids in local.items():
        #         # cluster2pid[i] += list(local[i])
        #         cluster2pid[idx] += list(pids)

        merging_way = self.config.ivf_merging_ways
        expected_merging_it = int(np.ceil(np.log(self.num_chunks)/np.log(merging_way)))
        Run().print_main(f"{merging_way}-way merging, expecting {expected_merging_it} iterations")

        candidate_fns = _balance_file_size(Path(working_dir).glob("localivf.0.*.pkl"))
        merging_it = 1
        while len(candidate_fns) > 1:
            n_merging = int(np.ceil(len(candidate_fns)/merging_way))
            Run().print_main(f"Merging IVF Iteration #{merging_it}: {len(candidate_fns)} files -- doing {n_merging} merging")

            work_ = partial(merge_ivf, working_dir, merging_it)
            with Pool(min(self.config.ivf_num_processes, n_merging)) as pool:
                list(pool.map(work_, enumerate(more_itertools.batched(candidate_fns, merging_way))))

            candidate_fns = _balance_file_size(Path(working_dir).glob(f"localivf.{merging_it}.*.pkl"))
            merging_it += 1

        # read final one
        Run().print_main("Read final IVF file")
        cluster2pid = [ [] for _ in range(self.num_partitions) ]
        merged_fn = candidate_fns[0]
        for idx, pids in _load_file(merged_fn).items():
            cluster2pid[idx] = list(pids)


        Run().print_main(f"#> Creating ivf tensor...")
        ivf_lengths = torch.tensor([ len(c) for c in cluster2pid ])
        ivf = torch.zeros(ivf_lengths.sum(), dtype=torch.int32)
        offset = 0
        for pids in cluster2pid:
            ivf[offset: offset+len(pids)] = torch.tensor(sorted(list(pids)))
            offset += len(pids)

        optimized_ivf_path = os.path.join(self.config.index_path_, 'ivf.pid.pt')
        torch.save((ivf, ivf_lengths), optimized_ivf_path)
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")

        # clean up 
        for fn in tqdm.tqdm(list(Path(working_dir).glob("localivf.*.pkl")), desc='clean up local ivf files'):
            fn.unlink()

    def _legacy_build_ivf(self):
        Run().print_main("#> Building IVF with lagacy build ivf code...")
        
        codes = torch.zeros(self.num_embeddings,).long()
        print_memory_stats(f'RANK:{self.rank}')

        Run().print_main("#> Loading codes...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(self.config.index_path_, chunk_idx)

            codes[offset:offset+chunk_codes.size(0)] = chunk_codes

        assert offset+chunk_codes.size(0) == codes.size(0), (offset, chunk_codes.size(0), codes.size())

        Run().print_main(f"Sorting codes...")

        print_memory_stats(f'RANK:{self.rank}')

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        print_memory_stats(f'RANK:{self.rank}')

        Run().print_main(f"Getting unique codes...")

        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        print_memory_stats(f'RANK:{self.rank}')

        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(ivf, ivf_lengths, self.config.index_path_)


    def _update_metadata(self):
        config = self.config
        self.metadata_path = os.path.join(config.index_path_, 'metadata.json')
        Run().print("#> Saving the indexing metadata to", self.metadata_path, "..")

        with open(self.metadata_path, 'w') as f:
            d = {'config': config.export()}
            d['num_chunks'] = self.num_chunks
            d['num_partitions'] = self.num_partitions
            d['num_embeddings'] = self.num_embeddings
            d['avg_doclen'] = self.num_embeddings / len(self.collection)

            f.write(ujson.dumps(d, indent=4) + '\n')


def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, shared_lists, return_value_queue=None):
    use_gpu = torch.cuda.is_available()
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)

    sample = shared_lists[0][0]
    sample = sample.float().numpy()

    kmeans.train(sample)

    centroids = torch.from_numpy(kmeans.centroids)

    print_memory_stats(f'RANK:0*')

    if return_value_queue is not None:
        return_value_queue.put(centroids)

    return centroids


# hack the multiprocessing here
# it really should be tagged along with the infrastructure
def build_local_ivf(index_path, working_dir, num_partitions, info):
    pid_start, local_doclen, chunk_idx = info

    output_fn = os.path.join(working_dir, f"localivf.0.{chunk_idx}.pkl")
    if os.path.exists(output_fn):
        Run().print_main(f"#> Found local ivf for chunk {chunk_idx}, skip")
        return 

    local_pid = 0
    passage_offset_end = local_doclen[0]
    cluster2pid = [ set() for _ in range(num_partitions) ]

    chunk_codes = ResidualCodec.Embeddings.load_codes(index_path, chunk_idx)
    for i, clusterid in enumerate(chunk_codes):
        if i >= passage_offset_end:
            local_pid += 1
            passage_offset_end += local_doclen[local_pid]
            
        cluster2pid[clusterid].add(int(pid_start + local_pid))

    # save to disk to avoid slow IPC
    cluster2pid = { cid: pids for cid, pids in enumerate(cluster2pid) if len(pids) > 0 }
    
    pickle.dump(cluster2pid, open(output_fn, 'wb'))
    Run().print_main(f"#> Chunk {chunk_idx} done")

    # return cluster2pid

def _balance_file_size(fns):
    fns = sorted(fns, key=lambda f: Path(f).stat().st_size)
    balanced = list(more_itertools.interleave_longest(fns[0::2], fns[1::2][::-1]))
    assert len(fns) == len(balanced)
    return balanced


def _load_file(fn):
    if str(fn).endswith('.pt'):
        return torch.load(fn)
    elif str(fn).endswith('.pkl'):
        return pickle.load(open(fn, 'rb'))

def merge_ivf(working_dir, current_it, info):
    running_id, fns = info

    cluster2pid = { cid: list(pids) for cid, pids in _load_file(fns[0]).items() }
    for fn in fns[1:]:
        for idx, pids in _load_file(fn).items():
            if idx not in cluster2pid:
                cluster2pid[idx] = []
            cluster2pid[idx] += list(pids)
    
    pickle.dump(cluster2pid, (Path(working_dir) / f"localivf.{current_it}.{running_id}.pkl").open('wb'))

    if current_it > 1:
        for fn in fns:
            Path(fn).unlink()

    Run().print_main(f"#> Merging id #{running_id} of iteration {current_it} done")

"""
TODOs:

1. Notice we're using self.config.bsize.

2. Consider saving/using heldout_avg_residual as a vector --- that is, using 128 averages!

3. Consider the operations with .cuda() tensors. Are all of them good for OOM?
"""

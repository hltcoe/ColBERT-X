import os
import argparse

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

from colbert.scripts.collection_utils import load_irds_or_local

def create_plaid_index(args):

    if args.n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = args.n_gpus

    # Provide v1 checkpoint configurations
    config = ColBERTConfig(
        doc_maxlen=args.doc_maxlen, nbits=args.nbits,
        max_sampled_pid=args.max_sampled_pid, max_num_partitions=args.max_num_partitions
    )

    collection = load_irds_or_local(
        args.dataset_name if args.coll_dir is None else str(os.path.join(args.coll_dir, 'collection_passages.tsv')),
        component='docs', use_offsetmap=args.lazy_collection_loader
    )
    dataset_name = args.dataset_name.replace('/', '.')

    if args.base_model is not None:
        config.configure(model_name=args.base_model)
    if args.checkpoint.endswith('.dnn'):
        # specifically for our ColBERT-X checkpoints
        config.configure(force_resize_embeddings=True, mask_punctuation=False)

    index_name = args.index_name or f'{dataset_name}.{args.nbits}bits'
    config.configure(
        index_name=index_name, 
        use_lagacy_build_ivf=args.use_lagacy_build_ivf,
        reuse_centroids_from=args.reuse_centroids_from
    )

    run_config = RunConfig(
        nranks=n_gpus, root=args.root, 
        ivf_num_processes=args.ivf_num_processes, 
        ivf_use_tempdir=args.ivf_use_tempdir, 
        ivf_merging_ways=args.ivf_merging_ways,
        experiment=args.experiment, 
        index_root=args.index_root
    )

    with Run().context(run_config):  # nranks specifies the number of GPUs to use.
        indexer = Indexer(checkpoint=args.checkpoint, config=config)

        if args.step == 'prepare':
            indexer.prepare(name=index_name, collection=collection, overwrite='resume')
        elif args.step == 'encode':
            indexer.encode(name=index_name, collection=collection)
        elif args.step == 'finalize':
            indexer.finalize(name=index_name, collection=collection)
            print(indexer.get_index()) # You can get the absolute path of the index, if needed.
            print("Index created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coll_dir', help="directory containing collection_passages.tsv file", default=None)
    parser.add_argument('--index_name', help="name of the index, default {dataset_name}.{nbits}bits", type=str, default=None)
    parser.add_argument('--dataset_name', help='name of the dataset', required=True)

    parser.add_argument('--step', help="step of indexing", choices=['prepare', 'encode', 'finalize'], required=True)
    parser.add_argument('--nbits', help='number of bits used to compress index', type=int, default=2)
    parser.add_argument('--doc_maxlen', help="the number used to truncate passages", type=int, default=300)

    parser.add_argument('--checkpoint', help="path to the model checkpoint", type=str, required=True)
    parser.add_argument('--base_model', help="base model of the v1 checkpoint", type=str, default=None)

    parser.add_argument('--index_root', help="directory where index is written", type=str, default=None)
    parser.add_argument('--experiment', help="name of the experiment", type=str, default="default")
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--n_gpus', help="number of GPU, default using all available ones", type=int, default=None)

    parser.add_argument('--max_num_partitions', help="max number of partitions/centroid used in indexing", type=int, default=-1)
    parser.add_argument('--max_sampled_pid', help="max number of sampled tokens for training KMeans", type=int, default=-1)

    parser.add_argument('--lazy_collection_loader', action='store_true', default=False)
    parser.add_argument('--ivf_num_processes', type=int, default=20)
    parser.add_argument('--ivf_use_tempdir', action='store_true', default=False)
    parser.add_argument('--ivf_merging_ways', type=int, default=2)

    parser.add_argument('--use_lagacy_build_ivf', action='store_true', default=False)    
    parser.add_argument('--use_offset_map', action='store_true', default=False)

    parser.add_argument('--reuse_centroids_from', type=str, default=None)

    create_plaid_index(parser.parse_args())
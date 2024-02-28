import argparse

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
from colbert.utils.utils import print_message

from colbert.scripts.collection_utils import load_irds_or_local

def main(args):
    bsize = args.per_device_batch_size*args.n_gpus
    exp_name = f"{args.run_tag}/{bsize}bat.{args.nway}way{'-ib' if args.ib_negatives else ''}"

    if args.use_local_full_triples:
        queries = None
        collection = None
    elif args.training_queries is not None:
        queries = load_irds_or_local(args.training_queries, 'queries')
        collection = load_irds_or_local(args.training_collection, 'docs', mixing=args.training_collection_mixing, use_offsetmap=args.use_offsetmap)
    else:
        queries = load_irds_or_local(args.training_irds_id, 'queries')
        collection = load_irds_or_local(args.training_irds_id, 'docs', use_offsetmap=args.use_offsetmap)

    triples = load_irds_or_local(args.training_triples, 'docpairs')

    with Run().context(RunConfig(nranks=args.n_gpus, root=args.root, experiment=args.experiment, name=exp_name)):
        trainer = Trainer(
            triples=triples, queries=queries, collection=collection,
            config=ColBERTConfig(
                bsize=bsize, 
                checkpoint=args.checkpoint,
                maxsteps=args.maxsteps, lr=args.learning_rate, nway=args.nway, 
                model_name=args.model_name, 
                similarity=args.similarity,
                force_resize_embeddings=True,
                use_ib_negatives=args.ib_negatives,
                ignore_scores=(args.kd_loss == 'None'),
                kd_loss=args.kd_loss, 
                shuffle_passages=not args.only_top, 
                sampling_max_beta=args.sampling_max_beta,
                over_one_epoch=True,
                resume=args.resume,
                resume_optimizer=args.resume_optimizer,
                fix_broken_optimizer_state=args.fix_broken_optimizer_state
            )
        )

        trainer.train()

        print_message(f"Best model: {trainer.best_checkpoint_path()}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--use_local_full_triples', action='store_true', default=False)
    parser.add_argument('--training_triples', type=str, required=True) # can be a irds id or path
    parser.add_argument('--training_irds_id', type=str, default='beir/msmarco/train')
    parser.add_argument('--training_queries', type=str, default=None)
    parser.add_argument('--training_collection', type=str, nargs='+', default=None)

    parser.add_argument('--use_offsetmap', action='store_true', default=False)

    parser.add_argument('--similarity', choices=['l2', 'cosine'], default='cosine')

    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--nway', type=int, default=8)
    parser.add_argument('--per_device_batch_size', type=int, default=8)
    parser.add_argument('--maxsteps', type=int, default=200_000)
    parser.add_argument('--ib_negatives', action='store_true', default=False)

    parser.add_argument('--kd_loss', choices=['KLD', 'MSE', 'None'], default='KLD')

    parser.add_argument('--only_top', action='store_true', default=False)
    parser.add_argument('--sampling_max_beta', type=float, default=1.0)

    parser.add_argument('--experiment', help="name of the experiment for locating the index", type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--n_gpus', help="number of gpus to use, default all visible ones in CUDA_VISIBLE_DEVICES", type=int, default=None)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_optimizer', action='store_true', default=False)
    parser.add_argument('--fix_broken_optimizer_state', action='store_true', default=False)

    args = parser.parse_args()

    if args.n_gpus is None:
        import torch
        args.n_gpus = torch.cuda.device_count()

    # make sure when running distillation, the triple file is a jsonl file
    # if args.kd_loss != 'None':
    #     assert args.training_triples.endswith('.jsonl')

    if args.training_queries is not None or args.training_collection is not None:
        assert args.training_queries is not None
        assert args.training_collection is not None
        print_message(f"Use separate irds for training queries and documents.")

    args.run_tag = args.run_tag or f"{args.learning_rate}"
    args.experiment = args.experiment or f"plaid_{args.model_name.split('/')[-1]}"

    main(args)
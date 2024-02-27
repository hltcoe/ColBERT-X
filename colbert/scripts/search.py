import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data.queries import Queries
from colbert import Searcher

from tqdm.auto import tqdm

import ir_measures as irms
from colbert.scripts.collection_utils import load_mapping

def maxp(passage_triples: List[Tuple[str, int, float]], mapping: Dict[str, str]) -> Dict[str, float]:
    doc_score = {}
    for pid, _, score in passage_triples:
        doc_id = mapping[pid]
        if doc_id not in doc_score or doc_score[doc_id] < score:
            doc_score[doc_id] = score
    return doc_score

def write_trec_run(doc_scores: Dict[str, Dict[str, float]], fn: str, run_id: str = 'PLAID', topk=1000):
    with open(fn, 'w') as fw:
        for qid, d in doc_scores.items():
            for i, (did, s) in enumerate(sorted(d.items(), key=lambda x:x[1], reverse=True)[:topk]):
                fw.write(f"{qid} 0 {did} {i+1} {s} {run_id}\n")

def main(args):
    output_fn = args.output or f"./{args.run_id}.{args.search_depth}.trec"
    if args.check_exists and Path(output_fn).exists():
        print(f"Found {output_fn} -- skip")
        return

    queries = Queries(path=args.query_file)
    mapping = load_mapping(args.passage_mapping, is_dummy=not args.maxp)

    if args.run_evel:
        metrics = [ irms.parse_measure(m) for m in args.metrics ]
        qrels = list(irms.read_trec_qrels(args.qrel))

    with Run().context(RunConfig(nranks=args.n_gpus, root=args.root, experiment=args.experiment, index_root=args.index_root)):
        checkpoint = None
        if args.checkpoint_root is not None:
            checkpoint = args.checkpoint_root + "/" + ColBERTConfig.load_from_index( args.index_root + "/" + args.index_name ).checkpoint
        searcher = Searcher(index=args.index_name, checkpoint=checkpoint)
        searcher.config.configure(only_approx=args.only_approx, ignore_unrecognized=False)
        
        rankings = searcher.search_all(queries, k=args.search_depth)

    doc_ranking = { 
        str(qid): maxp(r, mapping) for qid, r in tqdm(rankings.items(), desc='MaxP') 
    }

    if not args.no_save:
        print(f"Saving results to {output_fn}")
        write_trec_run(doc_ranking, output_fn, args.run_id)

    if args.run_evel:
        print( irms.calc_aggregate(metrics, qrels, doc_ranking) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', help="name of the index", type=str)
    parser.add_argument('--passage_mapping', help="mapping tsv file from passage to documents", type=str)
    parser.add_argument('--query_file', help="query tsv file", type=str)
    parser.add_argument('--search_depth', help="k; depth of PLAID search", type=int, default=2500)
    parser.add_argument('--centroid_score_threshold', help="threshold for centroid cell filter", type=float, default=0.4)

    parser.add_argument('--maxp', action='store_true', default=False)
    parser.add_argument('--checkpoint_root', type=str, default=None)

    parser.add_argument('--metrics', help="evalutation metrics used after search, should be parsable by ir_measures", 
                        type=str, nargs='*', default=[])
    parser.add_argument('--qrel', help="path to qrels file", type=str, default=None)

    parser.add_argument('--run_id', help="name of this run, will be used for the filename of the trec file", type=str, default=None)
    parser.add_argument('--output', help="output directory of the trec file", type=str, default=None)

    parser.add_argument('--index_root', help="root directory of the indexes", type=str, default=None)
    parser.add_argument('--experiment', help="name of the experiment for locating the index", type=str, default="default")
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--n_gpus', help="number of gpus to use, default all visible ones in CUDA_VISIBLE_DEVICES", type=int, default=None)

    parser.add_argument('--only_approx', action='store_true', default=False)
    parser.add_argument('--no_save', action='store_true', default=False)

    parser.add_argument('--check_exists', action='store_true', default=False)

    parser.set_defaults(run_evel=False)

    args = parser.parse_args()

    if args.n_gpus is None:
        import torch
        args.n_gpus = torch.cuda.device_count()
    
    if len(args.metrics) > 0:
        assert args.qrel and Path(args.qrel).exists(), args.qrel
        args.run_evel = True

    assert Path(args.query_file).exists(), args.query_file

    if args.run_id is None:
        args.run_id = f"{args.experiment}.{args.index_name}.{Path(args.query_file).name}"
        args.run_id = args.run_id.replace('/', '.')

    main(args)
import os
import argparse
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that generates inverted index from a jsonl index file")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--mapping", type=str, required=True)
    parser.add_argument("--rank_file", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="xlmr")
    parser.add_argument("--qry_map", type=str, help="mapping file for query ids if needed")

    args = parser.parse_args()

    mapping_dict = {}
    with open(args.mapping) as f:
        for line in f:
           map_id, pass_id = line.strip().split("\t")
           mapping_dict[map_id] = pass_id

    qry_map_dict = {}
    if args.qry_map:
        with open(args.qry_map) as fin:
            for line in fin:
                map_id, qry_id = line.strip().split("\t")
                qry_map_dict[map_id] = qry_id
    
    rank_dict = defaultdict(list)
    with open(args.rank_file) as f:
        for line in f:    
            lsplit = line.strip().split("\t")
            qid = lsplit[0]
            if qid in qry_map_dict:
                qid = qry_map_dict[qid]
            map_id = lsplit[1]
            score = lsplit[-1]
            rank_dict[qid].append((mapping_dict[map_id], score))

    agg_dict = defaultdict(list)
    seen_pairs = defaultdict(int)
    cutoff = 1000
    for qid, pass_list in rank_dict.items():
        rank = 1
        for pass_id, score in pass_list:
            doc_id = pass_id.split("_")[0]
            if (qid, doc_id) not in seen_pairs:
                agg_dict[qid].append(doc_id)
                seen_pairs[(qid, doc_id)] = score
                if rank==cutoff: break
                rank += 1

                

    fname = os.path.join(args.root, f"{args.prefix}_ranking.trec")
    with open(fname, "w") as f:
        for qid, doc_list in agg_dict.items():
            for rank, doc_id in enumerate(doc_list, start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{seen_pairs[(qid,doc_id)]}\tColXLMR\n")


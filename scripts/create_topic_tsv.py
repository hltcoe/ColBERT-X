import os
import re
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that generates topic tsv file from a query jsonl file")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--field", type=str, default="topic_title")
    parser.add_argument("--qid", type=str, default="topic_id")
    parser.add_argument("--map_qid", help='map query_ids to ints', action='store_true')
    parser.add_argument("--clef", help='process originals to match qrels', action='store_true')
    parser.add_argument("--truecase", help='keep the true case of the topics', action='store_true')

    args = parser.parse_args()

    qid_map = {}
    filename = f"{args.split}_{args.field}.tsv"
    if args.truecase:
        filename = f"{args.split}_{args.field}_cased.tsv"
    fname = os.path.join(args.root, filename)
    with open(args.query_file) as f, open(fname, "w") as g:
        for i, line in enumerate(f):
            tmp = json.loads(line)
            qid, qtext = tmp[args.qid], tmp[args.field]
            if args.map_qid:
                qid_map[i+1] = qid
                qid = i+1
            if not args.truecase:
                qtext = qtext.lower()
            g.write(f"{qid}\t{qtext}\n")

    if args.map_qid:
        fname = os.path.join(args.root, f"{args.split}_{args.field}.tsv-mapping.txt")
        with open(fname, 'w') as fout:
            for new_id, orig_id in qid_map.items():
                if (args.clef and orig_id.startswith('C')):
                    orig_id = int(orig_id[1:])
                print(f'{new_id}\t{orig_id}', file=fout)

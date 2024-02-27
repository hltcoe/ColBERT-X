from pathlib import Path
from typing import Dict, List, Union

import gzip
import pickle
import json
from time import time
import re
from tqdm import tqdm

from colbert.infra import Run, ColBERTConfig
from colbert.utils.utils import print_message, easy_pbar
from colbert.data import Queries, Collection, Examples

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import ir_datasets as irds

import logging

_component_cls_map: Dict[str, Union[Examples, Queries, Collection]] = { 
    'docpairs': Examples, 'queries': Queries, 'docs': Collection 
}


def load_mapping(f: str, is_dummy: bool=False) -> Dict[int, str]:
    return { 
        int(pid): "_".join(tag.split("_")[:-1]) if not is_dummy else tag
        for pid, tag in map(lambda l: l.strip().split("\t"), open(f)) 
    }


def strip_newlines(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


class _irds_wrapper:
    def __init__(self, dsid, component='docs'):
        self.dsid = dsid
        self.component = component

        self.id_key = 'doc_id' if component == 'docs' else 'query_id'
    
    @property
    def data(self):
        return getattr(irds.load(self.dsid), self.component)

    def _process(self, d):
        return d.text

    def _get_id(self, d):
        return getattr(d, self.id_key)

    def __getitem__(self, i):
        if isinstance(i, int):
            i = str(i)
        return self._process(self.data.lookup(i))
    
    def __iter__(self):
        return map(self._process, iter(self.data))

    def __len__(self):
        return len(self.data)
    
    def items(self):
        yield from ( (self._get_id(d), self._process(d)) for d in self.data )


class OffsetMapCollection(Collection):
    def __init__(self, config: ColBERTConfig, collection_path: str, offsetmap_path: str=None, build_offset: bool=True):
        self.config = config
        assert collection_path.endswith('.tsv') or collection_path.endswith('.tsv.gz'), "Only support tsv for now"
        self.collection_path = collection_path

        if build_offset:
            self.path, self.mapping = self._build_or_load_offset_map(collection_path, offsetmap_path)
            self.length = len(self.mapping)
        else:
            self.length = self._count_num_lines()
            self.path, self.mapping = None, None
        
        self._reader = None
    
    def _get_opener(self):
        return (gzip.open if self.collection_path.endswith('.gz') else open)

    @property
    def reader(self):
        # avoid the file pointer being pickled
        if self._reader is None or self._reader.closed:
            self._reader = (gzip.open if self.collection_path.endswith('.gz') else open)(self.collection_path, 'rt')
        return self._reader

    def _count_num_lines(self):
        return sum(1 for _ in easy_pbar(self._get_opener()(self.collection_path, 'rt'), "counting length", disabled=False, use_tqdm=True))

    def _load_offset_map(self, offsetmap_path: str, collection_path: str):
        if not offsetmap_path:
            return None
        
        target_path, mapping =  pickle.load(open(offsetmap_path, 'rb'))
        assert Path(target_path).absolute() == Path(collection_path).absolute(), f"Mapping was built for {target_path} not {collection_path}."
        return str(offsetmap_path), mapping

    def _build_or_load_offset_map(self, collection_path: str, offsetmap_path: str=None):
        if offsetmap_path is None:
            index_path = Path(self.config.index_path_)
            index_path.mkdir(parents=True, exist_ok=True)
            offsetmap_path = str(index_path / "offset_map.pkl")

        if Path(offsetmap_path).exists():
            return self._load_offset_map(offsetmap_path, collection_path)

        mapping = []
        
        # explicitly avoid touching self.reader
        with self._get_opener()(self.collection_path, 'rt') as fr:
            for i, _ in enumerate(easy_pbar(desc='building offset map', disabled=False, use_tqdm=True)): # while True
                loc = fr.tell()
                line = fr.readline()
                if line == '':
                    break
                assert int(line.split('\t')[0]) == i
                mapping.append(loc)
        
        print(f"> Writing offset map to {offsetmap_path}")
        with open(offsetmap_path, 'wb') as fw:
            pickle.dump((str(collection_path), mapping), fw)
            fw.flush()
        
        return offsetmap_path, mapping

    def _parse_line(self, line: str):
        pid, passage, *rest = line.strip('\n\r ').split('\t')
        return (rest[0] + ' | ' + passage) if len(rest) >= 1 else passage
        
    def __iter__(self):
        self.reader.seek(0)
        yield from map(
            self._parse_line, 
            easy_pbar(self.reader, desc="reading collection on-the-fly", disabled=Run().config.rank != 0)
            # one process displaying the progress would be enough
        )

    def __getitem__(self, idx: Union[int, List[int]]):
        assert self.mapping is not None, f"Have not built the offset map."

        if isinstance(idx, list):
            return [ self.__getitem__(i) for i in idx ]

        self.reader.seek(self.mapping[int(idx)])
        return self._parse_line(self.reader.readline())

    def __len__(self):
        return self.length
    
    def save(self, new_path):
        raise NotImplementedError(f"Offset map collection is not meant to be stored. ")


def _load_single_collection(c: Union[str, Collection], use_offsetmap: bool=False):
    if isinstance(c, str) and use_offsetmap:
        print_message(f"Loading {c} with offsetmap...")
        p = Path(c)
        if (p.parent / f"{p.name}.offsetmap").exists():
            return OffsetMapCollection(
                collection_path=c, 
                offsetmap_path=str(p.parent / f"{p.name}.offsetmap"),
                config=None
            )
    return Collection.cast(c)


def load_irds_or_local(ds: str, component: str, mixing='elements', use_offsetmap=False) -> Union[Examples, Collection, Examples, str]:
    if ds in irds.registry:
        print_message(f"Use IRDS:{ds} for {component}")
        return _component_cls_map[component](data=_irds_wrapper(ds, component), path=f"irds:{ds}:{component}")
    else:
        print_message(f"Use local file (or hfhub if applicable):{ds} for {component}")
        # assert Path(ds).exists(), f"File {ds} does not exists."

        # avoid materialize it in the main process if using raw file
        return _load_single_collection(ds, use_offsetmap=True) if use_offsetmap else ds


def _offsetmap_cache_cli(args):
    collection_path: Path = args.collection_path
    offsetmap_path = collection_path.parent / f"{collection_path.name}.offsetmap"
    print_message(collection_path)
    OffsetMapCollection(None, str(collection_path), offsetmap_path, build_offset=True)


def _process_documents(args):
    if not Path(args.corpus).exists():
        try:
            args.corpus = hf_hub_download(*str(args.corpus).split(":"), repo_type="dataset")
        except EntryNotFoundError:
            raise FileNotFoundError(f"`{args.corpus}` does not exist on disk nor Huggingface Hub")

    opener = gzip.open if args.corpus.endswith('.gz') else open
    num_docs = sum(1 for _ in tqdm(opener(args.corpus, 'rt'), desc='counting'))
    with opener(args.corpus, 'rt') as f:
        for i, line in tqdm(enumerate(f), total=num_docs):
            try:
                temp = json.loads(line)
            except json.decoder.JSONDecodeError:
                print(f"json decode error on line #{i} -- `{line}`")
                continue
            doc_id = temp[args.docid]
            title = strip_newlines(temp[args.title]) if args.title in temp else ""
            text = strip_newlines(temp[args.body])
            doc_text = title + " " + text if title else text
            if args.lower:
                doc_text = doc_text.lower()
            encoded_tokens = args.tokenizer.encode(doc_text)[1:-1]
            s, e, idx = 0, 0, 0
            while s < len(encoded_tokens):
                e = s + args.length
                if e >= len(encoded_tokens):
                    e = len(encoded_tokens)
                p = args.tokenizer.decode(encoded_tokens[s:e])
                pass_id = f"{doc_id}_{idx}"
                s = s + args.length - args.stride
                yield p, pass_id
                idx += 1


def _create_passage_collection_cli(args):
    from transformers import AutoTokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) 

    root: Path = args.root
    if not root.exists():
        root.mkdir(parents=True)
    args.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pass_file= root / "collection_passages.tsv"
    map_file = root / "mapping.tsv"
    
    with open(pass_file, "w") as f, open(map_file, "w") as g:
        for idx,(line1,line2) in enumerate(_process_documents(args)):
            f.write(f"{idx}\t{line1}\n")
            g.write(f"{idx}\t{line2}\n")


def _merge_teacher_scores(args):
    assert len(args.input_files) > 1, "We can only merge >1 files"
    
    combined = {}
    
    for fn in tqdm(args.input_files, desc="reading file...", dynamic_ncols=True):
        with open(fn) as fr:
            for triple_list in tqdm(map(json.loads, fr), dynamic_ncols=True):
                qid = triple_list[0]
                assert isinstance(qid, str)
                if qid not in combined:
                    combined[qid] = []
                combined[qid] += triple_list[1:]
    
    with open(args.output_file, "w") as fw:
        for qid, triple_list in tqdm(combined.items(), total=len(combined), desc='writing...'):
            fw.write(json.dumps([qid] + triple_list) + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Collection helper cli")
    subparser = parser.add_subparsers()

    offsetmap_parser = subparser.add_parser("build_offsetmap")
    offsetmap_parser.add_argument('collection_path', type=Path)
    offsetmap_parser.set_defaults(func=_offsetmap_cache_cli)

    passaging_parser = subparser.add_parser("create_passage_collection")
    passaging_parser.add_argument("--root", type=Path, required=True)
    passaging_parser.add_argument("--corpus", type=Path, required=True)
    passaging_parser.add_argument("--length", type=int, default=180)
    passaging_parser.add_argument("--stride", type=int, default=90)
    passaging_parser.add_argument("--docid", type=str, default="id")
    passaging_parser.add_argument("--title", type=str, default="title")
    passaging_parser.add_argument("--body", type=str, default="text")
    passaging_parser.add_argument("--lower", action="store_true", default=False) 
    passaging_parser.add_argument('--tokenizer', type=str, default="xlm-roberta-large")
    passaging_parser.set_defaults(func=_create_passage_collection_cli)

    teacher_merging_parser = subparser.add_parser("merge_teacher_scores")
    teacher_merging_parser.add_argument("--input_files", nargs='+', type=Path, required=True)
    teacher_merging_parser.add_argument("--output_file", type=Path, required=True)
    teacher_merging_parser.set_defaults(func=_merge_teacher_scores)

    args = parser.parse_args()
    args.func(args)

import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
        The defaults here have a special status in Run(), which initially calls assign_defaults(),
        so these aren't soft defaults in that specific context.
    """

    debug: bool = DefaultVal(False)
    overwrite: bool = DefaultVal(False)

    root: str = DefaultVal(os.path.join(os.getcwd(), 'experiments'))
    experiment: str = DefaultVal('default')

    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))

    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)

    ivf_num_processes: int = DefaultVal(20)
    ivf_use_tempdir: bool = DefaultVal(False)
    ivf_merging_ways: int = DefaultVal(2)

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(',')

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, 'indexes/')

    @property
    def script_name_(self):
        if '__file__' in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd):]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath):]
                except:
                    pass


            assert script_path.endswith('.py')
            script_name = script_path.replace('/', '.').strip('.')[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return 'none'

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def checkpoint_path_(self):
        return os.path.join(self.path_, "checkpoints")

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = DefaultVal("[unused0]")
    doc_token_id: str = DefaultVal("[unused1]")
    query_token: str = DefaultVal("[Q]")
    doc_token: str = DefaultVal("[D]")


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)


@dataclass
class DocSettings:
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(220)
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens : bool = DefaultVal(False)
    interaction: str = DefaultVal('colbert')


@dataclass
class TrainingSettings:
    similarity: str = DefaultVal('cosine')

    bsize: int = DefaultVal(32)

    accumsteps: int = DefaultVal(1)

    lr: float = DefaultVal(3e-06)

    maxsteps: int = DefaultVal(500_000)

    save_every: int = DefaultVal(None)

    resume: bool = DefaultVal(False)

    resume_optimizer: bool = DefaultVal(False)

    fix_broken_optimizer_state: bool = DefaultVal(False)

    ## NEW:
    warmup: int = DefaultVal(None)

    warmup_bert: int = DefaultVal(None)

    relu: bool = DefaultVal(False)

    nway: int = DefaultVal(2)

    n_query_alternative: int = DefaultVal(1)

    use_ib_negatives: bool = DefaultVal(False)

    kd_loss: str = DefaultVal("KLD")

    reranker: bool = DefaultVal(False)

    distillation_alpha: float = DefaultVal(1.0)

    ignore_scores: bool = DefaultVal(False)

    model_name: str = DefaultVal("bert-base-uncased")

    force_resize_embeddings: bool = DefaultVal(True)

    shuffle_passages: bool = DefaultVal(False)

    sampling_max_beta: float = DefaultVal(1.0)

    over_one_epoch: bool = DefaultVal(False)

    multilang: bool = DefaultVal(False)

    nolangreg: bool = DefaultVal(False)

@dataclass
class IndexingSettings:
    index_path: str = DefaultVal(None)

    nbits: int = DefaultVal(1)

    kmeans_niters: int = DefaultVal(4)

    resume: bool = DefaultVal(False)

    max_sampled_pid: int = DefaultVal(-1)
    max_num_partitions: int = DefaultVal(-1)

    use_lagacy_build_ivf: bool = DefaultVal(False)

    reuse_centroids_from: str = DefaultVal(None)

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)

@dataclass
class SearchSettings:
    ncells: int = DefaultVal(None)
    centroid_score_threshold: float = DefaultVal(None)
    ndocs: int = DefaultVal(None)

    only_approx: bool = DefaultVal(False)
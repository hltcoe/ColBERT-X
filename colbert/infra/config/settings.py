import os
import torch

import __main__
from dataclasses import dataclass, field
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
        The defaults here have a special status in Run(), which initially calls assign_defaults(),
        so these aren't soft defaults in that specific context.
    """

    debug: bool = field(default_factory=lambda: DefaultVal(False))
    overwrite: bool = field(default_factory=lambda: DefaultVal(False))

    root: str = field(default_factory=lambda: DefaultVal(os.path.join(os.getcwd(), 'experiments')))
    experiment: str = field(default_factory=lambda: DefaultVal('default'))

    index_root: str = field(default_factory=lambda: DefaultVal(None))
    name: str = field(default_factory=lambda: DefaultVal(timestamp(daydir=True)))

    rank: int = field(default_factory=lambda: DefaultVal(0))
    nranks: int = field(default_factory=lambda: DefaultVal(1))
    amp: bool = field(default_factory=lambda: DefaultVal(True))

    ivf_num_processes: int = field(default_factory=lambda: DefaultVal(20))
    ivf_use_tempdir: bool = field(default_factory=lambda: DefaultVal(False))
    ivf_merging_ways: int = field(default_factory=lambda: DefaultVal(2))

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = field(default_factory=lambda: DefaultVal(RunSettings.total_visible_gpus))

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
    query_token_id: str = field(default_factory=lambda: DefaultVal("[unused0]"))
    doc_token_id: str = field(default_factory=lambda: DefaultVal("[unused1]"))
    query_token: str = field(default_factory=lambda: DefaultVal("[Q]"))
    doc_token: str = field(default_factory=lambda: DefaultVal("[D]"))


@dataclass
class ResourceSettings:
    checkpoint: str = field(default_factory=lambda: DefaultVal(None))
    triples: str = field(default_factory=lambda: DefaultVal(None))
    collection: str = field(default_factory=lambda: DefaultVal(None))
    queries: str = field(default_factory=lambda: DefaultVal(None))
    index_name: str = field(default_factory=lambda: DefaultVal(None))


@dataclass
class DocSettings:
    dim: int = field(default_factory=lambda: DefaultVal(128))
    doc_maxlen: int = field(default_factory=lambda: DefaultVal(220))
    mask_punctuation: bool = field(default_factory=lambda: DefaultVal(True))


@dataclass
class QuerySettings:
    query_maxlen: int = field(default_factory=lambda: DefaultVal(32))
    attend_to_mask_tokens: bool = field(default_factory=lambda: DefaultVal(False))
    interaction: str = field(default_factory=lambda: DefaultVal('colbert'))


@dataclass
class TrainingSettings:
    similarity: str = field(default_factory=lambda: DefaultVal('cosine'))
    bsize: int = field(default_factory=lambda: DefaultVal(32))
    accumsteps: int = field(default_factory=lambda: DefaultVal(1))
    lr: float = field(default_factory=lambda: DefaultVal(3e-06))
    maxsteps: int = field(default_factory=lambda: DefaultVal(500_000))
    save_every: int = field(default_factory=lambda: DefaultVal(None))
    resume: bool = field(default_factory=lambda: DefaultVal(False))
    resume_optimizer: bool = field(default_factory=lambda: DefaultVal(False))
    fix_broken_optimizer_state: bool = field(default_factory=lambda: DefaultVal(False))
    warmup: int = field(default_factory=lambda: DefaultVal(None))
    warmup_bert: int = field(default_factory=lambda: DefaultVal(None))
    relu: bool = field(default_factory=lambda: DefaultVal(False))
    nway: int = field(default_factory=lambda: DefaultVal(2))
    n_query_alternative: int = field(default_factory=lambda: DefaultVal(1))
    use_ib_negatives: bool = field(default_factory=lambda: DefaultVal(False))
    kd_loss: str = field(default_factory=lambda: DefaultVal("KLD"))
    reranker: bool = field(default_factory=lambda: DefaultVal(False))
    distillation_alpha: float = field(default_factory=lambda: DefaultVal(1.0))
    ignore_scores: bool = field(default_factory=lambda: DefaultVal(False))
    model_name: str = field(default_factory=lambda: DefaultVal("bert-base-uncased"))
    force_resize_embeddings: bool = field(default_factory=lambda: DefaultVal(True))
    shuffle_passages: bool = field(default_factory=lambda: DefaultVal(False))
    sampling_max_beta: float = field(default_factory=lambda: DefaultVal(1.0))
    over_one_epoch: bool = field(default_factory=lambda: DefaultVal(False))
    multilang: bool = field(default_factory=lambda: DefaultVal(False))
    nolangreg: bool = field(default_factory=lambda: DefaultVal(False))


@dataclass
class IndexingSettings:
    index_path: str = field(default_factory=lambda: DefaultVal(None))
    nbits: int = field(default_factory=lambda: DefaultVal(1))
    kmeans_niters: int = field(default_factory=lambda: DefaultVal(4))
    resume: bool = field(default_factory=lambda: DefaultVal(False))
    max_sampled_pid: int = field(default_factory=lambda: DefaultVal(-1))
    max_num_partitions: int = field(default_factory=lambda: DefaultVal(-1))
    use_lagacy_build_ivf: bool = field(default_factory=lambda: DefaultVal(False))
    reuse_centroids_from: str = field(default_factory=lambda: DefaultVal(None))

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    ncells: int = field(default_factory=lambda: DefaultVal(None))
    centroid_score_threshold: float = field(default_factory=lambda: DefaultVal(None))
    ndocs: int = field(default_factory=lambda: DefaultVal(None))
    only_approx: bool = field(default_factory=lambda: DefaultVal(False))

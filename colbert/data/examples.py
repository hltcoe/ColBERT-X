from colbert.infra.run import Run
import os
import gzip
import ujson

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from colbert.utils.utils import print_message, easy_pbar
from colbert.utils.save_metadata import get_metadata_only

from colbert.infra.provenance import Provenance


class Examples:
    def __init__(self, path=None, data=None, nway=None, provenance=None):
        self.__provenance = provenance or path or Provenance()
        self.nway = nway
        self.path = path
        self.data = data or self._load_file(path)

    def provenance(self):
        return self.__provenance
    
    def toDict(self):
        return self.provenance()

    def _load_file(self, path: str):
        nway = self.nway + 1 if self.nway else self.nway
        examples = []

        if not os.path.exists(path):
            try:
                path = hf_hub_download(*path.split(":"), repo_type="dataset")
            except EntryNotFoundError:
                raise FileNotFoundError(f"`{path}` does not exist on disk nor Huggingface Hub")

        with (gzip.open if path.endswith('.gz') else open)(path, 'rt') as f:
            it = easy_pbar(f, desc=f'Loading {path}', disabled=Run().config.rank != 0)
            if path.endswith('.tsv'):
                for i, line in enumerate(it):
                    examples.append(line.strip().split("\t")[:nway])
                    if Run().config.debug and i > 10000:
                        break
            else:
                for line in it:
                    example = ujson.loads(line)[:nway]
                    examples.append(example)

        return examples

    def tolist(self, rank=None, nranks=None):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """

        if rank or nranks:
            assert rank in range(nranks), (rank, nranks)
            return [self.data[idx] for idx in range(rank, len(self.data), nranks)]  # if line_idx % nranks == rank

        return list(self.data)

    def save(self, new_path):
        assert 'json' in new_path.strip('/').split('/')[-1].split('.'), "TODO: Support .json[l] too."

        print_message(f"#> Writing {len(self.data) / 1000_000.0}M examples to {new_path}")

        with Run().open(new_path, 'w') as f:
            for example in self.data:
                ujson.dump(example, f)
                f.write('\n')

            output_path = f.name
            print_message(f"#> Saved examples with {len(self.data)} lines to {f.name}")
        
        with Run().open(f'{new_path}.meta', 'w') as f:
            d = {}
            d['metadata'] = get_metadata_only()
            d['provenance'] = self.provenance()
            line = ujson.dumps(d, indent=4)
            f.write(line)

        return output_path

    @classmethod
    def cast(cls, obj, nway=None):
        if type(obj) is str:
            return cls(path=obj, nway=nway)

        if isinstance(obj, list):
            return cls(data=obj, nway=nway)

        # if type(obj) is cls:
        if isinstance(obj, cls):
            # assert nway is None, nway
            assert obj.nway == nway, nway
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

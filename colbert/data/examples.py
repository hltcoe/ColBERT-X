from colbert.infra.run import Run
import os
import ujson

from colbert.utils.utils import print_message


class Examples:
    def __init__(self, path=None, data=None):
        self.path = path
        self.data = data or self._load_file(path)
    
    def provenance(self):
        return self.path

    def _load_file(self, path):
        examples = []
        
        with open(path) as f:
            for line in f:
                examples.append(ujson.loads(line))

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
            return [self.data[idx] for idx in range(0, len(self.data), nranks)] # if line_idx % nranks == rank

        return list(self.data)

    def save(self, new_path):
        assert 'json' in new_path.strip('/').split('/')[-1].split('.'), "TODO: Support .json[l] too."

        print_message(f"#> Writing {len(self.data) / 1000_000.0}M examples to {new_path}")

        with Run().open(new_path, 'w') as f:
            for example in self.data:
                ujson.dump(example, f)
                f.write('\n')

            return f.name
        # print_message(f"#> Saved ranking of {len(self.data)} queries and {len(self.flat_ranking)} lines to {new_path}")

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if isinstance(obj, list):
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"
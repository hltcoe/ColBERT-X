# PLAID-X

This is a generalized version of [PLAID](https://github.com/stanford-futuredata/ColBERT) and the previous ColBERT-X for CLIR.
The codebase supports models trained with the original ColBERT-X scripts, which are not compatible with the PLAID codebase released from the Stadford Futuredata Group. 

## Resources

We release a set of CLIR models in our [Translate-Distill](https://huggingface.co/collections/hltcoe/translate-distill-659a11e0a7f2d2491780a6bb) and [Multilingual Translate-Distill Huggingface Space](https://huggingface.co/collections/hltcoe/multilingual-translate-distill-66280df75c34dbbc1708a22f). 
Feel free to try it out! 

## Installation

PLAID-X is available on PyPi. You can install it through
```bash
pip install PLAID-X
```

Make sure your gcc and gxx version is `>=9.4.0`, which is the requirement for `ninja` to work properly.
We recommend using a `conda` environment to control it.

To support GPU for fitting the K-Means clustering during PLAID-X indexing, you can install the GPU extra with
```bash
pip install PLAID-X[gpu]
```


## Usage

We have published a [tutorial](https://github.com/hltcoe/clir-tutorial) on CLIR with notebooks to run various models. 
Please refer to the [PLAID-X notebook](https://colab.research.google.com/github/hltcoe/clir-tutorial/blob/main/notebooks/clir_tutorial_plaidx.ipynb) there for a simple working example in Python. 

The following provides a series of CLI commands for running a larger scale. 

### Training Models

The following command starts the training process using the `t53b-monot5-msmarco-engeng.jsonl.gz` triple file on the Huggingface Dataset repository [`hltcoe/tdist-msmarco-scores`](https://huggingface.co/datasets/hltcoe/tdist-msmarco-scores) with English queries and translated Chinese passages from [neuMARCO](https://ir-datasets.com/neumarco.html).

```bash
python -m colbert.scripts.train \
--model_name xlm-roberta-large \
--training_triples hltcoe/tdist-msmarco-scores:t53b-monot5-msmarco-engeng.jsonl.gz \
--training_irds_id neumarco/zh/train \
--maxsteps 200000 \
--learning_rate 5e-6 \
--kd_loss KLD \
--only_top \
--per_device_batch_size 8 \
--nway 6 \
--run_tag test \
--experiment test
```

For training MLIR models using Multilingual Translate-Distill, pass more multiple dataset ids to `--training_irds_id` flag along with a `--training_collection_mixing` for the mixing strategies (one of `entries`, `passages`, or `round-robin`). 
For more details, please read our paper **Distillation for Multilingual Information Retrieval**(arxiv link TBD).

### Indexing 

Since PLAID-X is a passage retrieval engine, you need to create passage collections if you are intended to search a document collection.
The following command creates a passage collection for the NeuCLIR1 Chinese corpus (file implicitly downloaded from Huggingface). 

```bash
python -m colbert.scripts.collection_utils create_passage_collection \
--root ./test_coll/ --corpus neuclir/neuclir1:data/zho-00000-of-00001.jsonl.gz
```

The indexing processes is broken into **three** steps. 
This is changed from the last version where we have two steps and also different from the original Stanford codebase where they combines everything into one Python call.
Separating the steps provides better allocation for the computation resources and avoid bad GPU reservation deadlocks between Pytorch and FAISS.

```bash
for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir ./test_coll \
--index_name test_index \
--dataset_name test_coll \
--nbits 1 \
--step $step \
--checkpoint eugene-yang/plaidx-xlmr-large-mlir-neuclir \
--experiment test 
done
```
Note that the `--checkpoint` flag accept ColBERT-X and ColBERT models stored on Huggingface Models.

### Searching 

Finally, the following command searches the collection with a query `.tsv` file where the first column is the query id and the second column contains the query text. 

```bash
python -m colbert.scripts.search \
--index_name neuclir-zho.1bits \
--passage_mapping ./test_coll/mapping.tsv \
--query_file query.tsv \
--metrics nDCG@20 MAP R@100 R@1000 Judged@10 \
--qrel qrels.txt \
--experiment test
```

### PLAID SHIRTTT Indexing

For replicating PLAID SHIRTTT experiments, we have released the 
[date of each document in NeuCLIR1 and ClueWeb09 on Huggingface](https://huggingface.co/datasets/hltcoe/plaid-shirttt-doc-date). 
To combine the ranks lists from each shard, you can use the following utility script to do so. 

```bash
python -m colbert.scripts.shirttt_utils --input {ranking files from each shard} --output {file to write} --topn 50 
```

## Citation and Credit

Please cite the following paper if you use the CLIR generalization of ColBERT.
```bibtex
@inproceedings{ecir2022colbert-x,
	author = {Suraj Nair and Eugene Yang and Dawn Lawrie and Kevin Duh and Paul McNamee and Kenton Murray and James Mayfield and Douglas W. Oard},
	title = {Transfer Learning Approaches for Building Cross-Language Dense Retrieval Models},
	booktitle = {Proceedings of the 44th European Conference on Information Retrieval (ECIR)},
	year = {2022},
	url = {https://arxiv.org/abs/2201.08471}
}
```

Please cite the following paper if you use the **MLIR** generalization. 
```bibtex
@inproceedings{ecir2023mlir,
	title = {Neural Approaches to Multilingual Information Retrieval},
	author = {Dawn Lawrie and Eugene Yang and Douglas W Oard and James Mayfield},
	booktitle = {Proceedings of the 45th European Conference on Information Retrieval (ECIR)},
	year = {2023},
	url = {https://arxiv.org/abs/2209.01335}
}
```

Please cite the following paper if you use the PLAID-X updated implemention or the translate-distil capability of the codebase. 
```bibtex
@inproceedings{ecir2024translate-distill,
  author = {Eugene Yang and Dawn Lawrie and James Mayfield and Douglas W. Oard and Scott Miller},
  title = {Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation},
  booktitle = {Proceedings of the 46th European Conference on Information Retrieval (ECIR)},
  year = {2024},
  url = {https://arxiv.org/abs/2401.04810}
}
```

Please cite the following paper if you use **Multilingual Translate-Distill** to train MLIR model. 
```bibtex
@inproceedings{sigir2024shirttt,
  author = {Eugene Yang and Dawn Lawrie and James Mayfield},
  title = {Distillation for Multilingual Information Retrieval},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)},
  year = {2024}
}
```

Please cite the following paper if you use PLAID SHIRTTT. 
```bibtex
@inproceedings{sigir2024shirttt,
  author = {Dawn Lawrie and Efsun Kayi and Eugene Yang and James Mayfield and Douglas W. Oard},
  title = {PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)},
  year = {2024}
}
```

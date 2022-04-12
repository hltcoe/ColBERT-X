# ColBERT-X

### ColBERT-X is a generalization of [ColBERT](https://github.com/stanford-futuredata/ColBERT) for cross-language retrieval. 

## Training

ColBERT-X can be trained in two ways, 
* Zero-Shot (ZS) using English MS MARCO triples, and 
* Translate-Train (TT) using translated MS MARCO triples.
The command for training is shown below:

```
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.run --nproc_per_node=4 -m \
xlmr_colbert.train --amp --doc_maxlen 180 --bsize 128 --accum 1 \
--triples /path/to/MSMARCO/triples.train.small.tsv --maxsteps 200000 \
--root /root/to/experiments/ --experiment MSMARCO-CLIR --similarity l2 --run msmarco.clir.l2
```


Detailed instructions for inference and PRF coming soon!

## Changelog

Here we list the differences between the ColBERT v1 codebase and our code
* Changed the model prefix from bert to roberta. Relevant issue [here](https://github.com/stanford-futuredata/ColBERT/issues/12#issuecomment-873047396). This is necessary as the incorrect model prefix will not let the pretrained model weights be loaded and they would be initialized from scratch.
* \<PAD\> token id is 0 for bert tokenizer and 1 for roberta tokenizer. Relevant line [here](https://github.com/hltcoe/ColBERT-X/blob/main/xlmr_colbert/modeling/colbert.py#L71).
* roberta tokenizer does not include additional '\[unused\]' token prefix in the vocabulary. So, they have to be manually added and the embeddings have to be resized. [Reference](https://github.com/stanford-futuredata/ColBERT/issues/12#issue-752674636)  

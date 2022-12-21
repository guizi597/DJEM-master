## DJME

This is code for DJME(Dynamically Joined Multiaspect Embeddings for Entity Alignment)

### Dependencies

- Python 3 (tested on 3.6.9)
- Pytorch (tested on 1.1.0)
- [transformers](https://github.com/huggingface/transformers) (tested on 2.1.1)
- Numpy

### How to Run

The model runs in four steps:

#### 1. Fine-tune Basic BERT Unit

To fine-tune the Basic BERT Unit, use: 

```shell
cd bert_embedding_unit/
python main.py
```

The obtained Basic BERT Unit and some other data will be stored in:  `../Save_model`

#### 2. Run GCN Embedding unit

To run the GCN Embedding unit, use:

```shell
cd ../gcn_embedding_unit/
python main.py
```

The obtained GCN Embedding unit and some other data will be stored in:  `../graph_ckpt`

#### 3. Run Predicate Embedding unit

To run the Predicate Embedding unit 

```shell
cd ../predicate_embedding_unit/
python main.py
```

The obtained Predicate Embedding unit and some other data will be stored in:  `../predicate_ckpt`

#### 4. Run Sinkhorn alignment unit

First, dynamically weight each embedding, and then use the Sinkhorn algorithm to achieve entity alignment, use:

```shell
cd ../sinkhorn_alignment_unit/
python -u Critic_method.py
python -u get_entity_embedding.py
python -u entity_alignment.py
```

Note that `sinkhorn_alignment_unit/Param.py` is the config file.
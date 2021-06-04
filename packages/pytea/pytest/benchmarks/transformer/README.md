# Movie Ratings with BERT

This repository is a simplified version of [huggingface/transformers](https://github.com/huggingface/transformers)

We injected a wrong shape configuration at line 181 of [modeling_bert.py](./transformers/modeling_bert.py)

If you comment that line and uncomment line 185, the analyzer will not show an error.

## How to test

```bash
python bin/pytea.py experiment/transformer
```

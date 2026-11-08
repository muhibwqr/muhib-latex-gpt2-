# muhib-gpt2

GPT-2 (124M) running inside TeX. You compile a LaTeX document, and `pdflatex` itself tokenizes the prompt, runs all 12 transformer layers, and picks the next token, over and over, until the page is written.

```latex
\usepackage{muhib-gpt2}
...
\muhib[12]{Hello, my name is Muhib}
```

renders as

> **Hello, my name is Muhib**i. I am a student of the art of the sword

These are the same 12 tokens the float32 reference picks greedily for that prompt.

## What actually runs where

- **At compile time, only TeX runs.** The math is eTeX `\numexpr` integer arithmetic. It covers token and position embeddings, layer norm, the QKV projections, causal self-attention with a KV cache, softmax, GELU, the MLP, the final layer norm, the tied 50,257-way output head, and argmax. The byte-level BPE tokenizer and the decoder are TeX macros too. `compile.sh` passes `-no-shell-escape`, and nothing uses Lua.
- **Python runs only once, ahead of time.** It downloads the checkpoint and writes the weights out as plain-text `.tex` files (about 1.4 GB). It also holds the float reference (`ref.py`) and a bit-exact simulator of the TeX arithmetic (`sim.py`), which I used to check the TeX output layer by layer.

## Numbers

- Weights are quantized per row to 12-bit integers (11-bit for the output head), with fixed-point activations (scale 2^10).
- Each generated token costs about 124 million multiply-adds, all inside `\numexpr`.
- The full demo takes about 9 minutes to compile, roughly 30 s per token.
- Projected compile time, according to the author: instant.

## Run it

You need TeX Live (or MacTeX) with `pdflatex`, Python 3, and about 2 GB of disk.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py        # GPT-2 small from Hugging Face -> hf/
.venv/bin/python export.py          # weights -> tex/w/*.tex
.venv/bin/python export_vocab.py    # tokenizer -> tex/w/vocab.tex, merges.tex
.venv/bin/python export_mc.py 12    # cache-format copies of the layer matrices
tex/compile.sh                      # writes tex/muhib.pdf
```

`compile.sh` raises two of TeX's memory limits through environment variables (`save_size`, `extra_mem_top`). The default settings are too small: the vocabulary alone defines 50,257 control sequences.

## Speed

Most of the time goes into TeX tokenizing the weight text, not into the arithmetic. One benchmark on a 768x3072 matrix took 533 ms just to read and tokenize it, but only 146 ms to evaluate it once it was already in memory. So at startup TeX reads the matrices of the layers listed in `tex/w/cache.tex` once and stores each one as a macro, with every weight a single `\mathchardef` token. Every later forward step then skips re-reading those matrices. TeX's main memory tops out at roughly 186M words, which is enough for layers 0-4 (about 140M words). The other 7 layers and the output head still stream from disk. The full 12-token demo went from 453 s to 374-439 s across runs (all timed while screen recording), with identical token ids.

To check the arithmetic against Python, load the package with `\usepackage[debug]{muhib-gpt2}`. It prints per-layer checksums that should match `DEBUG=1 .venv/bin/python sim.py` exactly.

## Files

- `tex/muhib-gpt2.sty`: the model and tokenizer, as TeX macros
- `tex/muhib.tex`: the demo document
- `tex/compile.sh`: the compile wrapper
- `export.py`, `export_vocab.py`: write the weight and tokenizer files that TeX reads
- `sim.py`: integer simulator of the TeX arithmetic
- `ref.py`: float GPT-2 reference in NumPy

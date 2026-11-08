# muhib-gpt2

GPT-2 (124M) running inside TeX. You compile a LaTeX document, and `pdflatex` itself tokenizes the prompt, runs all 12 transformer layers, and picks the next token, over and over, until the page is written.

```latex
\usepackage{muhib-gpt2}
...
\muhib[6]{Hello, my name is Muhib}
```

renders as

> **Hello, my name is Muhib**i. I am a student

These are the same tokens the float32 reference picks greedily for that prompt (with 12 tokens it continues "of the art of the sword").

For every token, TeX also keeps the top 5 logits and runs an online softmax over all 50,257 of them. The log prints lines like `>>> GPT-2 token 1/7: " was" 17.8% (next best: " is" 15.9%, " contains" 5.4%)`, and the PDF gets one page per token with bars for the top 5 guesses. `top5_check.py` prints the same table from the Python simulator.

## What actually runs where

- **At compile time, only TeX runs.** The math is eTeX `\numexpr` integer arithmetic. It covers token and position embeddings, layer norm, the QKV projections, causal self-attention with a KV cache, softmax, GELU, the MLP, the final layer norm, the tied 50,257-way output head, and argmax. The byte-level BPE tokenizer and the decoder are TeX macros too. `compile.sh` passes `-no-shell-escape`, and nothing uses Lua.
- **The weight conversion can run in TeX too.** `tex/convert.tex` reads the raw `model.safetensors` with pdfTeX's `\pdffiledump` and writes the same `.tex` weight files as the Python exporter, byte for byte (see below). Python is otherwise used once, ahead of time: it downloads the checkpoint and can also write the weights out as plain-text `.tex` files (about 1.4 GB). It also holds the float reference (`ref.py`) and a bit-exact simulator of the TeX arithmetic (`sim.py`), which I used to check the TeX output layer by layer.

## Numbers

- Weights are quantized per row to 12-bit integers (11-bit for the output head), with fixed-point activations (scale 2^10).
- Each generated token costs about 124 million multiply-adds, all inside `\numexpr`.
- Live mode (`\usepackage[live]{muhib-gpt2}`, used by `muhib.tex`): each `pdflatex` run generates one more token and saves the tokens plus the attention KV cache to `muhib.gpt`, so the PDF grows by one word per compile. Delete `muhib.gpt` to start over. Measured: the first compile (11-token prompt + 1 token) takes about 134 s, then about 17-18 s per compile/token; the 7 tokens are " was published in the journal Nature." (ids 373, 3199, 287, 262, 3989, 10362, 13), the same as float GPT-2.
- Weights are streamed in 128-wide input chunks. Bytes 128-255 are active characters `\countdef`'d to the chunk's input registers, so a multiply-add costs TeX about 5 characters (`+1046*` plus one byte) to read. That roughly halved the reading cost (25 s down to about 17.6 s per token) with bit-identical checksums.
- In TeXstudio, set Build & View (`Tools\Commands\quick` in texstudio.ini) to repeat `txs:///pdflatex | txs:///view-pdf-internal --embedded` 8 times; the preview then refreshes after every compile.
- Projected compile time, according to the author: instant.

## Run it

You need TeX Live (or MacTeX) with `pdflatex`, Python 3, and about 2 GB of disk.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py        # GPT-2 small from Hugging Face -> hf/
.venv/bin/python export.py          # norms, embeddings -> tex/w/*.tex
.venv/bin/python export_act.py      # weight matrices in active-byte chunks -> tex/w/*.c<k>.tex
.venv/bin/python export_vocab.py    # tokenizer -> tex/w/vocab.tex, merges.tex
.venv/bin/python export_mc.py 12    # cache-format copies of the layer matrices
tex/compile.sh                      # writes tex/muhib.pdf
```

### Converting the checkpoint in TeX instead

Only `hf/model.safetensors`, `hf/vocab.json` and `hf/merges.txt` are needed (the official GPT-2 small files from Hugging Face). Then:

```sh
tex/convert.sh          # -> tex/w/, runs the conversion jobs in parallel pdflatex processes
# or, as one process:
cd tex && max_print_line=1000000 pdflatex -no-shell-escape convert.tex
```

`\pdffiledump` returns the checkpoint's bytes as hex. TeX parses the safetensors JSON header for tensor offsets, decodes each IEEE-754 float32 with `\numexpr`, quantizes each row exactly (round half to even on the exact value of w*qmax/amax), and writes the norms, embeddings, active-byte layer chunks, output head, vocabulary and merges. On a 12-core M4 Pro VM, `convert.sh` (12 pdflatex processes, one per job such as one layer matrix or one output-head slice) takes 89 s, and all 780 files it writes are byte-identical to the Python export. The shell script only starts the processes; each one is `pdflatex convert.tex` with `\cvjob{fc}{3}`-style job selection. Conv1D matrices are stored inputs-major, so for each output file TeX reads a 160-row x 128-input block with 128 `\\pdffiledump` calls, after a first pass over whole stored rows finds each row's max. `\def\cvout{dir/}` before `\input convert` changes the output directory. It writes the default-format files only, not the `compact` or `fonthbox` ones.

`compile.sh` raises two of TeX's memory limits through environment variables (`save_size`, `extra_mem_top`). The default settings are too small: the vocabulary alone defines 50,257 control sequences.

## TeXstudio

Opening `muhib.tex` in TeXstudio and pressing Build & View works when TeXstudio is started with the same environment variables (`env save_size=5000000 extra_mem_top=240000000 texstudio muhib.tex`). The PDF opens in the built-in viewer. TeXstudio's Messages pane only fills in after the compile finishes, so to watch tokens appear, run `tail -f muhib.log` in its Terminal tab.

## Speed

Most of the time goes into TeX tokenizing the weight text, not into the arithmetic. One benchmark on a 768x3072 matrix took 533 ms just to read and tokenize it, but only 146 ms to evaluate it once it was already in memory. So at startup TeX reads the matrices of the layers listed in `tex/w/cache.tex` once and stores each one as a macro, with every weight a single `\mathchardef` token. Every later forward step then skips re-reading those matrices. TeX's main memory tops out at roughly 186M words, which is enough for layers 0-4 (about 140M words). The other 7 layers and the output head still stream from disk. The full 12-token demo went from 453 s to 374-439 s across runs (all timed while screen recording), with identical token ids. Pruning does not help: zeroing even the smallest 50% of each row changes the output into "in, and I'm the name of the name. I" (`prune_test.py`).

Experimental `compact` option (`\usepackage[live,compact]{muhib-gpt2}`, files from `export_wb.py`): each layer weight becomes one byte, an active character `\countdef`'d to a register holding a 5-bit weight, so a multiply-add is 4 characters (`+`, weight byte, `*`, input byte). On an M4 Pro VM it cut a generation compile from 19.7 s to 16.5 s and a prompt compile from 13.4 s to 10.6 s. The 84 per-layer checksums match `QMAX=31 DEBUG=1 .venv/bin/python sim.py`, and both demo prompts keep their output, but 5-bit weights change the output on 4 of 8 other test prompts, so it is off by default.

Experimental `fonthbox` option (`\usepackage[live,fonthbox]{muhib-gpt2}`, files and fonts from `export_fh.py`): each 5-bit weight value is its own font whose glyph widths are weight times an 8-bit activation level, and each row chunk is typeset as an `\hbox`, so TeX's box packing sums the products in C. Five outlier dimensions of the final hidden state are handled with exact integer arithmetic in the output head. On an M4 Pro VM a generated token took about 6.6 s (19.7 s by default). The 84 per-layer checksums match `simfh.py`, and the demo sentence is unchanged, but it is a different quantization: it matches the default on 5 of 10 test prompts (8 of 10 match float GPT-2 for all 7 tokens, against 7 of 10 for the default).

To check the arithmetic against Python, load the package with `\usepackage[debug]{muhib-gpt2}`. It prints per-layer checksums that should match `DEBUG=1 .venv/bin/python sim.py` exactly.

## Files

- `tex/muhib-gpt2.sty`: the model and tokenizer, as TeX macros
- `tex/muhib.tex`: the demo document
- `tex/compile.sh`: the compile wrapper
- `export.py`, `export_vocab.py`: write the weight and tokenizer files that TeX reads
- `export_wb.py`: 5-bit weight-byte layer files for the experimental `compact` option
- `export_fh.py`, `simfh.py`: weight files, fonts and simulator for the experimental `fonthbox` option
- `tex/convert.tex`: the checkpoint-to-TeX converter written in TeX
- `sim.py`: integer simulator of the TeX arithmetic
- `ref.py`: float GPT-2 reference in NumPy

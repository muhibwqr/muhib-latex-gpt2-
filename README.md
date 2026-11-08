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

For every token, TeX also keeps the top 5 logits and runs an online softmax over all 50,257 of them. The log prints lines like `>>> GPT-2 token 1/7: " was" 17.8% (next best: " is" 15.9%, " contains" 5.4%)`, and the PDF gets one page per token with bars for the top 5 guesses.

## What actually runs where

Everything is TeX. The repo has no Python, Lua, JavaScript or shell scripts.

- **Compiling runs the model.** The math is eTeX `\numexpr` integer arithmetic. It covers token and position embeddings, layer norm, the QKV projections, causal self-attention with a KV cache, softmax, GELU, the MLP, the final layer norm, the tied 50,257-way output head, and argmax. The byte-level BPE tokenizer and the decoder are TeX macros too. Compile with `-no-shell-escape`; nothing uses Lua.
- **Converting the weights is TeX too.** `tex/convert.tex` reads the raw `model.safetensors` with pdfTeX's `\pdffiledump` and writes the `.tex` weight files (see below). During development its output was checked byte for byte against a separate exporter, and the TeX arithmetic was checked layer by layer against a bit-exact simulator.

## Numbers

- Weights are quantized per row to 12-bit integers (11-bit for the output head), with fixed-point activations (scale 2^10).
- Each generated token costs about 124 million multiply-adds, all inside `\numexpr`.
- Live mode (`\usepackage[live]{muhib-gpt2}`, used by `muhib.tex`): each `pdflatex` run generates one more token and saves the tokens plus the attention KV cache to `muhib.gpt`, so the PDF grows by one word per compile. Delete `muhib.gpt` to start over. Measured: the first compile (11-token prompt + 1 token) takes about 134 s, then about 17-18 s per compile/token; the 7 tokens are " was published in the journal Nature." (ids 373, 3199, 287, 262, 3989, 10362, 13), the same as float GPT-2.
- Weights are streamed in 128-wide input chunks. Bytes 128-255 are active characters `\countdef`'d to the chunk's input registers, so a multiply-add costs TeX about 5 characters (`+1046*` plus one byte) to read. That roughly halved the reading cost (25 s down to about 17.6 s per token) with bit-identical checksums.
- In TeXstudio, set Build & View (`Tools\Commands\quick` in texstudio.ini) to repeat `txs:///pdflatex | txs:///view-pdf-internal --embedded` 8 times; the preview then refreshes after every compile.
- Projected compile time, according to the author: instant.

## Run it

You need TeX Live (or MacTeX) with `pdflatex`, `curl`, and about 2 GB of disk.

```sh
mkdir -p hf && for f in model.safetensors vocab.json merges.txt; do
  curl -L -o hf/$f https://huggingface.co/openai-community/gpt2/resolve/main/$f; done
cd tex
max_print_line=1000000 pdflatex -no-shell-escape convert.tex      # raw checkpoint -> tex/w/ (one process: ~14 min)
env save_size=5000000 extra_mem_top=240000000 pdflatex -no-shell-escape muhib.tex     # one token per run
env save_size=5000000 extra_mem_top=240000000 pdflatex -no-shell-escape muhibai.tex   # the demo video's prompt (15 runs)
```

Run the compile line again and again: each run adds one token to the PDF. Delete `muhib.gpt` (or `muhibai.gpt`) to start over. The two environment variables raise TeX's memory limits; the defaults are too small, since the vocabulary alone defines 50,257 control sequences.

Tested on macOS with MacTeX 2026 (pdfTeX 1.40.29). Linux TeX Live should work the same way but has not been tested. MiKTeX on Windows probably ignores the two memory environment variables. Times depend on the CPU: the numbers here come from an M4 Pro, and the output tokens are the same on any machine because all of the math is integer.

### How the conversion works

`\pdffiledump` returns the checkpoint's bytes as hex. TeX parses the safetensors JSON header for tensor offsets, decodes each IEEE-754 float32 with `\numexpr`, quantizes each row exactly (round half to even on the exact value of w*qmax/amax), and writes the norms, embeddings, active-byte layer chunks, output head, vocabulary and merges: 780 files. Conv1D matrices are stored inputs-major, so for each output file TeX reads a 160-row x 128-input block with 128 `\pdffiledump` calls, after a first pass over whole stored rows finds each row's max. `\def\cvout{dir/}` before `\input convert` changes the output directory.

The work splits into independent jobs (one layer matrix, one output-head slice, ...). `\def\cvjobs{\cvjob{fc}{3}}` before `\input convert` runs just one job, so several `pdflatex` processes can convert in parallel: on a 12-core M4 Pro VM, 12 processes took 89 s.

## TeXstudio

Opening `muhib.tex` in TeXstudio and pressing Build & View works when TeXstudio is started with the same environment variables (`env save_size=5000000 extra_mem_top=240000000 texstudio muhib.tex`). The PDF opens in the built-in viewer. TeXstudio's Messages pane only fills in after the compile finishes, so to watch tokens appear, run `tail -f muhib.log` in its Terminal tab.

## Speed

Most of the time goes into TeX tokenizing the weight text, not into the arithmetic. One benchmark on a 768x3072 matrix took 533 ms just to read and tokenize it, but only 146 ms to evaluate it once it was already in memory.

## Files

- `tex/muhib-gpt2.sty`: the model and tokenizer, as TeX macros
- `tex/convert.tex`: the checkpoint-to-TeX converter, in TeX
- `tex/muhib.tex`: the demo document
- `tex/muhibai.tex`: the prompt from the demo video

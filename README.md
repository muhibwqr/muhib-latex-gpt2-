# muhib-gpt2

GPT-2 (124M) running inside TeX. You compile a LaTeX document, and `pdflatex` itself tokenizes the prompt, runs all 12 transformer layers, and picks the next token, over and over, until the page is written.

```latex
\usepackage[live]{muhib-gpt2}
...
\muhib[9]{The most dangerous thing about AI is}
```

renders, after 15 compiles, as

> **The most dangerous thing about AI is** that it's so easy to make mistakes.

These are the same tokens float GPT-2 picks greedily for that prompt (see [Proof](#proof)).

For every token, TeX also keeps the top 5 logits and runs an online softmax over all 50,257 of them. The log prints lines like `>>> GPT-2 token 9/9: "." 46.8% (next best: "," 13.3%, " and" 9.1%)`, and the PDF gets one page per token with bars for the top 5 guesses.

## What actually runs where

Everything is TeX. The repo has no Python, Lua, JavaScript or shell scripts.

- **Compiling runs the model.** The math is eTeX `\numexpr` integer arithmetic. It covers token and position embeddings, layer norm, the QKV projections, causal self-attention with a KV cache, softmax, GELU, the MLP, the final layer norm, the tied 50,257-way output head, and argmax. The byte-level BPE tokenizer and the decoder are TeX macros too. Compile with `-no-shell-escape`; nothing uses Lua.
- **Converting the weights is TeX too.** `tex/convert.tex` reads the raw `model.safetensors` with pdfTeX's `\pdffiledump` and writes the `.tex` weight files (see below). During development its output was checked byte for byte against a separate exporter, and the TeX arithmetic was checked layer by layer against a bit-exact simulator.

## Numbers

- Weights are quantized per row to 12-bit integers (11-bit for the output head), with fixed-point activations (scale 2^10).
- Each generated token costs about 124 million multiply-adds, all inside `\numexpr`.
- Live mode (`\usepackage[live]{muhib-gpt2}`, used by `muhib.tex`): each `pdflatex` run generates one more token and saves the tokens plus the attention KV cache to `muhib.gpt`, so the PDF grows by one word per compile. Delete `muhib.gpt` to start over. The prompt is also read one token per compile, so `muhib.tex` ("Muhib wrote a paper about", 6 prompt tokens, 7 generated) takes 12 compiles: about 13.4 s per prompt token and 19.7 s per generated token. Its 7 tokens are " the issue in the journal Science." (ids 262, 2071, 287, 262, 3989, 5800, 13), the same as float GPT-2.
- Weights are streamed in 128-wide input chunks. Bytes 128-255 are active characters `\countdef`'d to the chunk's input registers, so a multiply-add costs TeX about 5 characters (`+1046*` plus one byte) to read. That roughly halved the reading cost (25 s down to about 17.6 s per token) with bit-identical checksums.
- In TeXstudio, set Build & View (`Tools\Commands\quick` in texstudio.ini) to repeat `txs:///pdflatex | txs:///view-pdf-internal --embedded` 8 times; the preview then refreshes after every compile.
- Projected compile time, according to the author: instant.

## Proof

**Same tokens as float GPT-2.** A float64 NumPy forward pass of the same checkpoint, picking greedily, gives exactly the token ids the TeX run gives:

| prompt (`tex/` file) | TeX output (ids) | float GPT-2 |
|---|---|---|
| "The most dangerous thing about AI is" (`muhibai.tex`, 15 compiles) | " that it's so easy to make mistakes." (326, 340, 338, 523, 2562, 284, 787, 10135, 13) | identical |
| "Muhib wrote a paper about" (`muhib.tex`, 12 compiles) | " the issue in the journal Science." (262, 2071, 287, 262, 3989, 5800, 13) | identical |

The last compile of the `muhibai.tex` run logged:

```text
[GPT-2 live: resuming at token 9, position 14] [GPT-2 prompt = 7 tokens]
>>> GPT-2 token 9/9: "." 46.8% (next best: "," 13.3%, " and" 9.1%)
```

**Same checkpoint.** These are the files `curl` downloads; the `model.safetensors` hash equals the `X-Linked-ETag` Hugging Face serves for it.

```text
248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707  model.safetensors
196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783  vocab.json
1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5  merges.txt
```

**Same converted weights.** After `pdflatex convert.tex`, all 780 files in `tex/w/` together hash to the value below, whether conversion runs in 1 process or 12:

```sh
cd tex/w && LC_ALL=C ls | LC_ALL=C xargs cat | shasum -a 256
# 2a55eea0c561c82fdbc8de400ad4fbc05c0f4cb2cb91e012f9a5a8591522d74e
```

During development the same 780 files were also byte-identical to a separate float exporter, and TeX's 84 per-layer checksums matched a bit-exact simulator of its integer math.

**Measured time** (M4 Pro VM, while screen recording): 14.3 s per prompt token and 20.8 s per generated token for `muhibai.tex`; conversion 92 s with 12 parallel `pdflatex` processes, 13 min 50 s with one.

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

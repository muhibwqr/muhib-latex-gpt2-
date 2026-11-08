"""Write GPT-2 weights as plain-text data files that TeX reads with \\input."""
import numpy as np, json, os, sys
from multiprocessing import Pool
from safetensors.numpy import load_file
from sim import quant_rows, S, L, HEAD_QMAX

OUT = 'tex/w'
LET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def nm(prefix, i): return '\\!' + prefix + LET[i // 52] + LET[i % 52]
A = [nm('', i) for i in range(768)]
G = [nm('G', i) if i < 2704 else nm('H', i - 2704) for i in range(3072)]
_W = None
def W():
    global _W
    if _W is None: _W = {k: v.astype(np.float64) for k, v in load_file('hf/model.safetensors').items()}
    return _W
fx = lambda a: np.round(a * 2**S).astype(np.int64)

def rows(path, w, bias, names, qmax=None):
    q, m = quant_rows(w) if qmax is None else quant_rows(w, qmax)
    b = fx(bias) if bias is not None else np.zeros(len(q), dtype=np.int64)
    with open(path, 'w') as f:
        for r in range(len(q)):
            nz = np.nonzero(q[r])[0]
            terms = ''.join(f'{q[r, i]:+d}*{names[i]}' for i in nz) or '0'
            f.write(f'\\!Y{{{m[r]}}}{{{b[r]}}}{terms}\\relax\n')
def lnfile(path, g, b):
    with open(path, 'w') as f:
        for gi, bi in zip(fx(g), fx(b)): f.write(f'\\!L{{{gi}}}{{{bi}}}\n')
def embfile(path, tab, start, var):
    with open(path, 'w') as f:
        for r in range(len(tab)):
            f.write(f'\\ifnum\\{var}={start + r} ' + ''.join(f'\\!x {v} ' for v in fx(tab[r])) + '\\fi\n')

def job(j):
    kind, a = j
    w = W()
    if kind == 'layer':
        p = f'h.{a}.'
        rows(f'{OUT}/l{a}qkv.tex', w[p+'attn.c_attn.weight'].T, w[p+'attn.c_attn.bias'], A)
        rows(f'{OUT}/l{a}aproj.tex', w[p+'attn.c_proj.weight'].T, w[p+'attn.c_proj.bias'], A)
        rows(f'{OUT}/l{a}fc.tex', w[p+'mlp.c_fc.weight'].T, w[p+'mlp.c_fc.bias'], A)
        rows(f'{OUT}/l{a}mproj.tex', w[p+'mlp.c_proj.weight'].T, w[p+'mlp.c_proj.bias'], G)
        lnfile(f'{OUT}/l{a}ln1.tex', w[p+'ln_1.weight'], w[p+'ln_1.bias'])
        lnfile(f'{OUT}/l{a}ln2.tex', w[p+'ln_2.weight'], w[p+'ln_2.bias'])
    elif kind == 'head':  # tied LM head, split in 8 parts; quantization is per-row so splitting is exact
        lo, hi = a * 6283, min(50257, (a + 1) * 6283)
        rows(f'{OUT}/head{a}.tex', w['wte.weight'][lo:hi], None, A, HEAD_QMAX)
    elif kind == 'wte':
        lo = a * 256
        embfile(f'{OUT}/wte{a}.tex', w['wte.weight'][lo:lo + 256], lo, 'gpt@tok')
    elif kind == 'wpe':
        lo = a * 256
        embfile(f'{OUT}/wpe{a}.tex', w['wpe.weight'][lo:lo + 256], lo, 'gpt@pos')
    elif kind == 'misc':
        lnfile(f'{OUT}/lnf.tex', w['ln_f.weight'], w['ln_f.bias'])
    return j

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = [('layer', l) for l in range(L)] + [('head', i) for i in range(8)] + \
           [('wte', c) for c in range(197)] + [('wpe', c) for c in range(4)] + [('misc', 0)]
    with Pool(8) as p:
        for j in p.imap_unordered(job, jobs): print('done', j, flush=True)

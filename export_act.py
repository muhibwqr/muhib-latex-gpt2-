"""Weight matrices in 'active byte' format: input i of each 128-wide chunk is
the byte 128+i, an active character \\countdef'd to the activation register.
Line = one row's chunk:  \\!S / \\!U / \\!Y{m}{b}  then  +w*<byte>...  then \\relax."""
import numpy as np, os
from multiprocessing import Pool
from safetensors.numpy import load_file
from sim import quant_rows, S, L, HEAD_QMAX
OUT = 'tex/w'; B = 128
fx = lambda a: np.round(a * 2**S).astype(np.int64)
_W = None
def W():
    global _W
    if _W is None: _W = {k: v.astype(np.float64) for k, v in load_file('hf/model.safetensors').items()}
    return _W
def rows(name, w, bias, qmax=None):
    q, m = quant_rows(w) if qmax is None else quant_rows(w, qmax)
    b = fx(bias) if bias is not None else np.zeros(len(q), dtype=np.int64)
    nch = q.shape[1] // B
    for c in range(nch):
        with open(f'{OUT}/{name}.c{c}.tex', 'wb') as f:
            blk = q[:, c*B:(c+1)*B]
            for r in range(len(q)):
                nz = np.nonzero(blk[r])[0]
                t = b''.join(b'%+d*' % blk[r, j] + bytes([128 + j]) for j in nz)
                hd = b'\\!Y{%d}{%d}' % (m[r], b[r]) if c == nch - 1 else (b'\\!S' if c == 0 else b'\\!U')
                f.write(hd + t + b'\\relax\n')
def job(j):
    kind, a = j; w = W()
    if kind == 'layer':
        p = f'h.{a}.'
        for n, k in (('qkv', 'attn.c_attn'), ('aproj', 'attn.c_proj'), ('fc', 'mlp.c_fc'), ('mproj', 'mlp.c_proj')):
            rows(f'l{a}{n}', w[p+k+'.weight'].T, w[p+k+'.bias'])
    else:
        lo, hi = a * 6283, min(50257, (a + 1) * 6283)
        rows(f'head{a}', w['wte.weight'][lo:hi], None, HEAD_QMAX)
    return j
if __name__ == '__main__':
    with Pool(8) as p:
        for j in p.imap_unordered(job, [('head', i) for i in range(8)] + [('layer', l) for l in range(L)]): print('done', j, flush=True)

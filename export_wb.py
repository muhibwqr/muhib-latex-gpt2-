"""Layer matrices in 'weight byte' format (5-bit weights, QMAX=31): term = +<w>*<x>.
<w> is one of 62 single-byte active characters \\countdef'd to a register holding the
weight (-31..31), <x> is the active input byte 128+i as in export_act.py.  Files: <name>.k<c>.tex"""
import numpy as np, os
from multiprocessing import Pool
import export_act as E
from sim import quant_rows, L
QMAX = 31; B = 128
SAFE = [c for c in list(range(1, 9)) + [11, 12] + list(range(14, 32)) + list(range(65, 91)) + list(range(97, 123))
        if chr(c) not in 'SUYrelax'][:2 * QMAX]
CH = {w: bytes([SAFE[w + QMAX if w < 0 else w + QMAX - 1]]) for w in range(-QMAX, QMAX + 1) if w}
def rows(name, w, bias):
    q, m = quant_rows(w, QMAX)
    b = E.fx(bias)
    nch = q.shape[1] // B
    for c in range(nch):
        with open(f'{E.OUT}/{name}.k{c}.tex', 'wb') as f:
            blk = q[:, c*B:(c+1)*B]
            for r in range(len(q)):
                t = b''.join(b'+' + CH[int(blk[r, j])] + b'*' + bytes([128 + j]) for j in np.nonzero(blk[r])[0])
                hd = b'\\!Y{%d}{%d}' % (m[r], b[r]) if c == nch - 1 else (b'\\!S' if c == 0 else b'\\!U')
                f.write(hd + t + b'\\relax\n')
def job(l):
    w = E.W(); p = f'h.{l}.'
    for n, k in (('qkv', 'attn.c_attn'), ('aproj', 'attn.c_proj'), ('fc', 'mlp.c_fc'), ('mproj', 'mlp.c_proj')):
        rows(f'l{l}{n}', w[p+k+'.weight'].T, w[p+k+'.bias'])
    return l
if __name__ == '__main__':
    with open(f'{E.OUT}/wbyte.tex', 'w') as f:
        f.write(','.join(str(c) for c in SAFE) + '\n')
    with Pool(6) as p:
        for l in p.imap_unordered(job, range(L)): print('done layer', l, flush=True)

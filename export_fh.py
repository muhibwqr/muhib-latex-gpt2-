"""Matrices for the 'fonthbox' option.  A row chunk is typeset as an \\hbox: byte 1 opens it, then for
each weight value v present, a font-switch byte (font gf<v>: glyph c is (c-128)*v sp wide) followed by
the input bytes 128+j whose weight is v.  TeX lets byte 128+j be the glyph of activation level
x_j*127/max, so the box width is the dot product.  End bytes: 2 first chunk, 3 middle, 4{m}{b} last.
Files: <name>.f<c>.tex plus fonts gfp<v>.tfm / gfm<v>.tfm."""
import numpy as np, os, struct
from multiprocessing import Pool
import export_act as E
from sim import quant_rows, L
B = 128
FB = [b for b in range(5, 128) if b not in (9, 10, 13, 32, 34, 35, 37, 45, 92, 94, 123, 125, 126, 127) and not 48 <= b <= 57]
QF = len(FB) // 2
WV = [v for v in range(-QF, QF + 1) if v]
FBY = {v: bytes([FB[i]]) for i, v in enumerate(WV)}
def fontname(v): return 'gf%s%d' % ('m' if v < 0 else 'p', abs(v))
def tfm(v, path):
    levs = [c - 128 for c in range(256)]
    ws = [0] + sorted({v * l for l in levs if l}); idx = {w: i for i, w in enumerate(ws)}
    ci = [(idx[v * l] << 24) if l else 0 for l in levs]
    lh, nw, np_ = 2, len(ws), 7
    lf = 6 + lh + 256 + nw + 3 + np_
    out = struct.pack('>12H', lf, lh, 0, 255, nw, 1, 1, 1, 0, 0, 0, np_) + struct.pack('>iI', 0, 16 << 20)
    out += b''.join(struct.pack('>I', c) for c in ci) + b''.join(struct.pack('>i', w) for w in ws)
    out += bytes(12) + bytes(4 * np_)
    assert len(out) == 4 * lf and nw <= 256
    open(path, 'wb').write(out)
OUTD = [496, 430, 36, 314, 374]  # ln_f outlier dims, added exactly in the head (byte 127)
def rows(name, w, bias, head=False):
    if head:
        qd, md = quant_rows(w, E.HEAD_QMAX); w = w.copy(); w[:, OUTD] = 0
    q, m = quant_rows(w, QF)
    b = E.fx(bias) if bias is not None else np.zeros(len(q), dtype=np.int64)
    nch = q.shape[1] // B
    for c in range(nch):
        blk = q[:, c*B:(c+1)*B]
        with open(f'{E.OUT}/{name}.f{c}.tex', 'wb') as f:
            for r in range(len(q)):
                row = blk[r]; t = bytearray(b'\x01')
                for v in np.unique(row[row != 0]):
                    t += FBY[int(v)] + bytes(128 + j for j in np.nonzero(row == v)[0])
                if head and c == nch - 1: t += b'\x7f{%d}' % m[r] + b''.join(b'{%d}' % qd[r, d] for d in OUTD) + b'{%d}' % md[r]
                else: t += b'\x02' if c == 0 else b'\x03' if c < nch - 1 else b'\x04{%d}{%d}' % (m[r], b[r])
                f.write(bytes(t) + b'\n')
def job(j):
    kind, a = j; w = E.W()
    if kind == 'layer':
        p = f'h.{a}.'
        for n, k in (('qkv', 'attn.c_attn'), ('aproj', 'attn.c_proj'), ('fc', 'mlp.c_fc'), ('mproj', 'mlp.c_proj')):
            rows(f'l{a}{n}', w[p+k+'.weight'].T, w[p+k+'.bias'])
    else:
        lo, hi = a * 6283, min(50257, (a + 1) * 6283)
        rows(f'head{a}', w['wte.weight'][lo:hi], None, head=True)
    return j
if __name__ == '__main__':
    assert QF == 49
    for v in WV: tfm(v, f'{E.OUT}/{fontname(v)}.tfm')
    import sys
    jobs = [('head', i) for i in range(8)] + ([] if 'headonly' in sys.argv else [('layer', l) for l in range(L)])
    with Pool(6) as p:
        for j in p.imap_unordered(job, jobs): print('done', j, flush=True)

"""Integer mirror of the 'fonthbox' option: rows are typeset as \\hbox'es whose glyph widths are
weight*level, so TeX's box packing does the multiply-adds.  Weights: QMAX=QF per row (one font per
value); activations: per-vector levels lev = x*127/max (\\numexpr rounding).  Run: python simfh.py"""
import os
QF = int(os.environ.get('QF', 49))
os.environ['QMAX'] = str(QF)
import numpy as np, sim
from sim import tdiv, tmuldiv, quant_rows, track
OUTD = [496, 430, 36, 314, 374]
def levels(x):
    mx = max(1, int(np.abs(x).max()))
    return tdiv(np.asarray(x, dtype=np.int64) * 127, mx), mx
def dq(acc, m, mx):
    u = tmuldiv(acc, m, 2**20); track('fh.u', u)
    return tmuldiv(u, np.full(np.shape(u), mx), 130048)
class Model(sim.Model):
    def __init__(self):
        super().__init__()
        self._dh = type('H', (), {'head': quant_rows(self.W['wte.weight'], sim.HEAD_QMAX)})()
        wz = self.W['wte.weight'].copy(); wz[:, OUTD] = 0
        self.head = quant_rows(wz, QF)
    def matvec(self, name, wqm, x):
        q, m, b = wqm
        lev, mx = levels(x)
        acc = (q * lev[None, :]).sum(1); track(name + '.acc', acc)
        y = dq(acc, m, mx) + b; track(name + '.out', y)
        return y
    def headlogits(self, h):
        if os.environ.get('HEADDEF'): return sim.Model.headlogits(self._dh, h)
        q, m = self.head; qd, md = self._dh.head
        e = (qd[:, OUTD] * h[None, OUTD]).sum(1); track('head.out', e)
        hz = h.copy(); hz[OUTD] = 0
        lev, mx = levels(hz)
        acc = (q * lev[None, :]).sum(1); track('head.acc', acc)
        return dq(acc, m, mx) + tmuldiv(e, md, 2**30)
if __name__ == '__main__':
    import tiktoken, json, sys
    enc = tiktoken.get_encoding('gpt2'); M = Model()
    P = sys.argv[1:] or ["Muhib wrote a paper about", "The meaning of life is", "Once upon a time,", "The best thing about LaTeX is",
         "Muhib compiled a LaTeX document, and the document", "Hello, my name is", "TeX is a typesetting system that",
         "The capital of France is", "In 1969, the first man", "My favorite food is"]
    print(json.dumps({'QF': QF, 'out': [sim.generate(M, enc.encode(p), 7) for p in P]}), flush=True)
    for k, v in sorted(sim.stats.items()): print(f'{k:16s} {v:>12d}  {v/sim.LIM:.4f}')

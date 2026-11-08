"""Integer fixed-point GPT-2 that mirrors the TeX arithmetic exactly (eTeX \numexpr semantics)."""
import numpy as np, math
from safetensors.numpy import load_file
S = 10            # activation fractional bits
QMAX = 2047       # weight quantization range
HEAD_QMAX = 1023  # LM head: smaller range keeps the 768-term sums far from 2^31
SK = 8            # key fractional bits (cached keys)
F = 1 << 16       # exp / probability unit
LIM = 2**31 - 1
import os
DEBUG = bool(os.environ.get('DEBUG'))
L, D, H, HD = 12, 768, 12, 64
stats = {}
def track(name, a):
    m = int(np.abs(a).max()) if np.size(a) else 0
    stats[name] = max(stats.get(name, 0), m)
    assert m <= LIM, (name, m)
def tdiv(a, b):
    """\numexpr a/b : round half away from zero (works on int64 numpy arrays / ints)."""
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    s = np.sign(a) * np.sign(b)
    q = (2 * np.abs(a) + np.abs(b)) // (2 * np.abs(b))
    return s * q
def tmuldiv(a, b, c):
    """\numexpr a*b/c with 64-bit intermediate.  Uses python ints when the product could exceed int64."""
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object); c = np.asarray(c, dtype=object)
    p = a * b
    f = np.frompyfunc(lambda p, c: (1 if (p >= 0) == (c > 0) else -1) * ((2 * abs(p) + abs(c)) // (2 * abs(c))), 2, 1)
    r = f(p, c)
    return int(r) if np.ndim(r) == 0 else np.asarray(r, dtype=np.int64)

def quant_rows(w, qmax=QMAX):  # w: (out, in) float -> int rows in [-qmax, qmax], m (scale*2^30)
    amax = np.abs(w).max(1); amax[amax == 0] = 1
    s = amax / qmax
    q = np.round(w / s[:, None]).astype(np.int64)
    m = np.round(s * 2**30).astype(np.int64)
    return q, m

class Model:
    def __init__(self):
        W = {k: v.astype(np.float64) for k, v in load_file('hf/model.safetensors').items()}
        self.W = W
        fx = lambda a, s=S: np.round(a * 2**s).astype(np.int64)
        self.wte = fx(W['wte.weight']); self.wpe = fx(W['wpe.weight'])
        self.layers = []
        for l in range(L):
            p = f'h.{l}.'; d = {}
            for n in ['ln_1', 'ln_2']:
                d[n] = (fx(W[p+n+'.weight']), fx(W[p+n+'.bias']))
            for n in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
                q, m = quant_rows(W[p+n+'.weight'].T)
                d[n] = (q, m, fx(W[p+n+'.bias']))
            self.layers.append(d)
        self.lnf = (fx(W['ln_f.weight']), fx(W['ln_f.bias']))
        self.head = quant_rows(W['wte.weight'], HEAD_QMAX)
    def matvec(self, name, wqm, x):
        q, m, b = wqm
        prods = q * x[None, :]
        track(name + '.prefix', np.cumsum(prods, 1))
        acc = prods.sum(1)
        y = tmuldiv(acc, m, 2**30) + b
        track(name + '.out', y)
        return y
    def ln(self, name, x, gb):
        g, b = gb
        mean = int(tdiv(x.sum(), D)); track(name + '.sum', np.cumsum(x))
        d = x - mean
        maxd = int(np.abs(d).max())
        k = 0
        while int(tdiv(maxd, 2 ** k)) > 1182: k += 1   # keeps every d^2/4^k <= ~2^30/768
        sq = tmuldiv(d, d, 4 ** k)
        track(name + '.sq', np.cumsum(sq))
        v1 = int(tdiv(sq.sum(), D))
        j = 0
        while v1 * 4 ** (j + 1) < 2**30: j += 1
        sdhi = math.isqrt(v1 * 4 ** j)      # = sd * 2^(j-k) in units 2^-S
        # out = d * g * 2^(j-k) / sdhi ; shift distributes onto g
        sh = j - k
        if sh >= 0: gj = g * 2 ** sh
        else: gj = tdiv(g, 2 ** (-sh))
        track(name + '.gj', gj)
        y = tmuldiv(d, gj, sdhi) + b
        track(name + '.out', y)
        return y
    @staticmethod
    def exp_neg(y):  # y >= 0 in units F -> e^-y in units F
        LN2 = 45426
        n = y // LN2  # TeX: computed via truncation emulation
        r = y - n * LN2
        t = F
        for i in range(8, 0, -1):
            t = F - int(tmuldiv(r, t, i * F))
        if n > 30: return 0
        return int(tdiv(t, 2 ** n))
    def gelu(self, x):
        out = np.empty_like(x)
        for i, xi in enumerate(x.tolist()):
            if xi >= 8 << S: out[i] = xi; continue
            if xi <= -(8 << S): out[i] = 0; continue
            xf = xi << (16 - S)
            x2 = int(tmuldiv(xf, xf, F)); x3 = int(tmuldiv(x2, xf, F))
            u = int(tmuldiv(xf + int(tmuldiv(x3, 2930, F)), 52293, F))   # 0.044715*F=2930.4, sqrt(2/pi)*F=52293.4
            e = self.exp_neg(2 * abs(u))
            th = int(tmuldiv(F - e, F, F + e))
            if u < 0: th = -th
            out[i] = int(tmuldiv(xi, F + th, 2 * F))
        return out
    def step(self, tok, pos, cache):
        x = self.wte[tok] + self.wpe[pos]
        for l, d in enumerate(self.layers):
            h = self.ln(f'ln1', x, d['ln_1'])
            qkv = self.matvec('qkv', d['attn.c_attn'], h)
            q, k, v = qkv[:D], qkv[D:2*D], qkv[2*D:]
            kq = tdiv(k, 2 ** (S - SK))
            cache[l].append((kq, v))
            o = np.zeros(D, dtype=np.int64)
            for hh in range(H):
                sl = slice(hh * HD, (hh + 1) * HD)
                scores = []
                for (kk, vv) in cache[l]:
                    pr = q[sl] * kk[sl]; track('score.prefix', np.cumsum(pr))
                    scores.append(int(tdiv(pr.sum(), 8 * 2 ** (S + SK - 16))))
                mx = max(scores)
                es = [self.exp_neg(mx - s) for s in scores]
                E = sum(es)
                ps = [int(tmuldiv(e, F, E)) for e in es]
                for p_, (kk, vv) in zip(ps, cache[l]):
                    o[sl] += tmuldiv(np.full(HD, p_), vv[sl], F)
            track('attn.o', o)
            x = x + self.matvec('aproj', d['attn.c_proj'], o); track('resid', x)
            h = self.ln('ln2', x, d['ln_2'])
            g = self.gelu(self.matvec('fc', d['mlp.c_fc'], h)); track('gelu', g)
            x = x + self.matvec('mproj', d['mlp.c_proj'], g); track('resid', x)
            if DEBUG: print(f'[dbg pos={pos} layer={l} sum={int(tdiv(x, 16).sum())} x0={x[0]} x767={x[767]}]')
        h = self.ln('lnf', x, self.lnf)
        q, m = self.head
        prods = q * h[None, :]; track('head.prefix', np.cumsum(prods, 1))
        logits = tmuldiv(prods.sum(1), m, 2**30)
        return int(np.argmax(logits)), logits, h

def generate(model, ids, n):
    cache = [[] for _ in range(L)]
    out = []
    seq = list(ids)
    for pos in range(len(ids) + n - 1):
        t, logits, _ = model.step(seq[pos], pos, cache)
        if pos >= len(ids) - 1:
            out.append(t); seq.append(t)
    return out

if __name__ == '__main__':
    import tiktoken, time
    enc = tiktoken.get_encoding('gpt2')
    M = Model()
    for prompt in ['The meaning of life is', 'Hello, my name is', 'TeX is a typesetting system that']:
        t0 = time.time()
        out = generate(M, enc.encode(prompt), 8)
        print(repr(prompt), '->', repr(enc.decode(out)), out, f'{time.time()-t0:.1f}s', flush=True)
    for k, v in sorted(stats.items()): print(f'{k:16s} {v:>12d}  {v/LIM:.4f}')

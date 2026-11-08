import sys, numpy as np, sim, tiktoken
P = float(sys.argv[1]); orig = sim.quant_rows; kept = [0, 0]
def pq(w, qmax=sim.QMAX):
    if P > 0:
        thr = np.quantile(np.abs(w), P, axis=1, keepdims=True)
        w = np.where(np.abs(w) < thr, 0.0, w)
    q, m = orig(w, qmax); kept[0] += np.count_nonzero(q); kept[1] += q.size
    return q, m
sim.quant_rows = pq
enc = tiktoken.get_encoding('gpt2'); M = sim.Model()
ids = enc.encode('Hello, my name is Muhib'); out = sim.generate(M, ids, 12)
print(P, f'kept {kept[0]/1e6:.1f}M of {kept[1]/1e6:.1f}M', out, repr(enc.decode(out)), flush=True)

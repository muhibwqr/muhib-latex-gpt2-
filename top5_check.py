# Reference for the TeX top-5 table: the same fixed-point logits as sim.py, softmaxed in float.
import numpy as np, sim, tiktoken, sys
enc = tiktoken.get_encoding('gpt2'); M = sim.Model()
ids = enc.encode(sys.argv[1] if len(sys.argv) > 1 else 'Hello, my name is Muhib'); n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cache = [[] for _ in range(sim.L)]; seq = list(ids)
for pos in range(len(ids) + n - 1):
    t, lg, _ = M.step(seq[pos], pos, cache)
    if pos >= len(ids) - 1:
        z = np.asarray(lg, dtype=np.float64) / 2**sim.S; p = np.exp(z - z.max()); p /= p.sum()
        top = np.argsort(-np.asarray(lg), kind='stable')[:5]
        print(t, [(int(i), repr(enc.decode([int(i)])), round(100 * p[i], 2)) for i in top], flush=True)
        seq.append(t)

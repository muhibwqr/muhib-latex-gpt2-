# How many LM-head rows survive a low-rank upper bound?  bound_i = C_i.(U h) + |resid_i| |h_perp|
import numpy as np, sim, tiktoken, sys, time
enc = tiktoken.get_encoding('gpt2'); M = sim.Model()
q, m = M.head; qf = q.astype(np.float64); mf = np.asarray(m, dtype=np.float64).reshape(-1)
H = []
for prompt in ['Hello, my name is Muhib', 'The meaning of life is']:
    ids = enc.encode(prompt); cache = [[] for _ in range(sim.L)]; seq = list(ids)
    for pos in range(len(ids) + 5):
        t, lg, h = M.step(seq[pos], pos, cache)
        if pos >= len(ids) - 1: H.append(np.asarray(h, dtype=np.float64)); seq.append(t)
print(len(H), 'h vectors', flush=True)
_, s, Vt = np.linalg.svd(qf - 0, full_matrices=False)
for r in (16, 32, 64, 128, 256):
    U = Vt[:r]; C = qf @ U.T; R = np.linalg.norm(qf - C @ U, axis=1)
    a = []; b = []
    for h in H:
        ex = (qf @ h) * mf / 2**30; L = ex.max()
        g = U @ h; hp = np.linalg.norm(h - U.T @ g)
        bd = (C @ g + R * hp) * mf / 2**30
        a.append(int((bd >= L).sum())); b.append(int((bd >= L - 16384).sum()))
    print(f'r={r}: argmax candidates {a}  within-16 {b}', flush=True)

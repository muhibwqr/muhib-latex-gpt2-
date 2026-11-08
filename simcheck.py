import sim, tiktoken, sys
enc = tiktoken.get_encoding('gpt2'); M = sim.Model()
for p, n in [('Muhib compiled a LaTeX document, and the document', 8), ('Hello, my name is Muhib and I', 9)]:
    out = sim.generate(M, enc.encode(p), n); print(repr(p), out, repr(enc.decode(out)), flush=True)

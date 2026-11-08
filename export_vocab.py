"""Tokenizer data for TeX: vocab (id -> byte codes) and merges (id pair -> rank)."""
import json
v = json.load(open('hf/vocab.json'))
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
inv = {i: s for s, i in v.items()}
with open('tex/w/vocab.tex', 'w') as f:
    for i in range(len(inv)):
        f.write('\\!t{' + ''.join(f'{u2b[ch]},' for ch in inv[i]) + '}\n' if i != 50256 else '\\!t{}\n')
merges = [l for l in open('hf/merges.txt').read().split('\n')[1:] if l]
with open('tex/w/merges.tex', 'w') as f:
    for a_b in merges:
        a, b = a_b.split(' ')
        f.write(f'\\!m{{{v[a]}}}{{{v[b]}}}\n')

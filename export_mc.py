"""Write in-memory copies of the layer matrices for TeX to cache at load time.

Weights become \\mathchardef tokens (\\!Wab = |w|) so each term is 4 tokens:
sign, weight, *, activation.  TeX reads these files once, keeps each matrix
as a macro, and every forward step after that skips re-tokenizing the text.
"""
import re, sys
from multiprocessing import Pool
LET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def nm(k): return '\\!W' + LET[k // 52] + LET[k % 52]
TERM = re.compile(r'([+-])(\d+)\*')
def conv(name):
    src = open(f'tex/w/{name}.tex').read()
    out = TERM.sub(lambda m: m.group(1) + nm(int(m.group(2))) + '*', src)
    with open(f'tex/w/{name}.mc.tex', 'w') as f:
        f.write('\\gpt@cache{' + name + '}{' + out + '}\n')
    return name
if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with open('tex/w/wdefs.tex', 'w') as f:
        for k in range(2048): f.write('\\mathchardef' + nm(k) + '=' + str(k) + '\n')
    names = [f'l{l}{m}' for l in range(n) for m in ['qkv', 'aproj', 'fc', 'mproj']]
    with Pool(8) as p: print(len(p.map(conv, names)), 'matrices')
    with open('tex/w/cache.tex', 'w') as f:
        f.write('\\gpt@input{wdefs}\n' + ''.join('\\gpt@input{' + x + '.mc}' + '\n' for x in names))

import numpy as np, json
from safetensors.numpy import load_file
W = {k: v.astype(np.float64) for k, v in load_file('hf/model.safetensors').items()}
L, D, H = 12, 768, 12
def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def forward(ids):
    T = len(ids)
    x = W['wte.weight'][ids] + W['wpe.weight'][:T]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for l in range(L):
        p = f'h.{l}.'
        h = ln(x, W[p+'ln_1.weight'], W[p+'ln_1.bias'])
        qkv = h @ W[p+'attn.c_attn.weight'] + W[p+'attn.c_attn.bias']
        q, k, v = np.split(qkv, 3, -1)
        q = q.reshape(T, H, 64).transpose(1, 0, 2); k = k.reshape(T, H, 64).transpose(1, 0, 2); v = v.reshape(T, H, 64).transpose(1, 0, 2)
        a = q @ k.transpose(0, 2, 1) / 8 + mask
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(T, D)
        x = x + o @ W[p+'attn.c_proj.weight'] + W[p+'attn.c_proj.bias']
        h = ln(x, W[p+'ln_2.weight'], W[p+'ln_2.bias'])
        x = x + gelu(h @ W[p+'mlp.c_fc.weight'] + W[p+'mlp.c_fc.bias']) @ W[p+'mlp.c_proj.weight'] + W[p+'mlp.c_proj.bias']
    x = ln(x, W['ln_f.weight'], W['ln_f.bias'])
    return x @ W['wte.weight'].T
if __name__ == '__main__':
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    for prompt in ['The meaning of life is', 'Hello, my name is', 'TeX is a typesetting system that']:
        ids = enc.encode(prompt)
        out = []
        for _ in range(8):
            nxt = int(forward(ids + out)[-1].argmax()); out.append(nxt)
        print(repr(prompt), ids, '->', repr(enc.decode(out)), out)

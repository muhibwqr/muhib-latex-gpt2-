import tiktoken, ref
enc = tiktoken.get_encoding('gpt2')
P = ['Hello, my name is Muhib and I', 'My name is Muhib. I am a student at the University of Waterloo, and I',
 'Muhib Waqar is a student at the University of Waterloo. He', 'Hi! My name is Muhib and I am',
 'Muhib compiled a LaTeX document, and the document', 'The University of Waterloo is', 'My name is Muhib, and I love',
 'Muhib is a student at the University of Waterloo. He is', 'Once upon a time, Muhib', 'Muhib built GPT-2 in TeX because he']
for p in P:
    ids = enc.encode(p); out = []
    for _ in range(10): out.append(int(ref.forward(ids + out)[-1].argmax()))
    print(len(ids), repr(p), '->', repr(enc.decode(out)), flush=True)

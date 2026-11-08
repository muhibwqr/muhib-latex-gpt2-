"""Fetch the GPT-2 small checkpoint and tokenizer files into hf/."""
from huggingface_hub import hf_hub_download
for f in ['config.json', 'model.safetensors', 'vocab.json', 'merges.txt']:
    hf_hub_download('openai-community/gpt2', f, local_dir='hf')

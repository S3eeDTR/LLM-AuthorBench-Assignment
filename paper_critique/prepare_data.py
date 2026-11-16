"""Fetch the public benchmark and validate the existing Assignment 1 split."""
from pathlib import Path
import hashlib
import urllib.request
from project_utils import DATA_DIR, load_data

DATA_URL = ('https://raw.githubusercontent.com/LLMauthorbench/LLMauthorbench/'
            '6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7/LLM-AuthorBench.json.zip')
SPLIT_URL = ('https://raw.githubusercontent.com/S3eeDTR/LLM-AuthorBench-Assignment/'
             '2956a0ad07f49529026cacd33728018d6ebf96ca/data/splits.csv')

def verify(path, expected, normalize_lines=False):
    content=path.read_bytes()
    if normalize_lines:
        content=content.replace(b'\r\n',b'\n')
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError(f'Hash verification failed: {path}')

def main():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    inputs=[
        (DATA_URL,'LLM-AuthorBench.json.zip','e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c',False),
        (SPLIT_URL,'splits.csv','8a3c8e2883844295fbddf5b78b96898478f606305972de4feb07a238254df5f4',True),
    ]
    for url,name,expected,normalize in inputs:
        target=DATA_DIR/name
        if target.exists():
            verify(target,expected,normalize)
        else:
            temporary=target.with_suffix(target.suffix+'.download')
            urllib.request.urlretrieve(url,temporary)
            verify(temporary,expected,normalize)
            temporary.replace(target)
        print('Verified:',name)
    print(load_data().groupby(['split','label']).size().unstack())

if __name__=='__main__':
    main()

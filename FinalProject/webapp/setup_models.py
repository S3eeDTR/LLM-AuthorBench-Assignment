"""Prepare local weights and tokenizer after cloning the source repository."""
import os
os.environ['USE_TF'] = '0'
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed', type=Path, required=True, help='Verified 50%% removal checkpoint.')
    parser.add_argument('--stripped', type=Path, required=True, help='Verified 100%% removal checkpoint.')
    parser.add_argument('--tokenizer-source', type=Path, help='Optional existing tokenizer/config folder for offline setup.')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    folder = here.parent / 'models'
    manifest = json.loads((here / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    expected = {entry['condition']: entry for entry in manifest['files'].values()}
    folder.mkdir(exist_ok=True)
    for condition, source in [('050', args.mixed), ('100', args.stripped)]:
        if not source.is_file(): parser.error(f'Checkpoint not found: {source}')
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
        if digest.hexdigest() != expected[condition]['sha256']:
            parser.error(f'{source.name} does not match the verified {condition} checkpoint. Check the model order.')
        target = folder / f'codet5_remove_{condition}.pt'
        if target.resolve() != source.resolve(): shutil.copy2(source, target)
        print('Verified:', target.name)
    assets = folder / 'tokenizer'
    assets.mkdir(exist_ok=True)
    if args.tokenizer_source:
        for path in args.tokenizer_source.iterdir():
            if path.is_file() and (path.suffix in ['.json', '.txt'] or path.name == 'spiece.model'):
                target = assets / path.name
                if target.resolve() != path.resolve(): shutil.copy2(path, target)
    else:
        from transformers import AutoTokenizer, T5Config
        revision = manifest['model_revision']
        AutoTokenizer.from_pretrained('Salesforce/codet5p-770m', revision=revision).save_pretrained(assets)
        T5Config.from_pretrained('Salesforce/codet5p-770m', revision=revision).save_pretrained(assets)
    assert (assets / 'config.json').exists(), 'Tokenizer folder must include the encoder config.json.'
    (folder / 'checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Models ready. Run START_WEB_APP.cmd or python webapp/server.py.')

if __name__ == '__main__': main()

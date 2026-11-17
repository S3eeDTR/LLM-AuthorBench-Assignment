"""Local inference with the verified course-project checkpoints."""
from pathlib import Path
import os
os.environ['USE_TF'] = '0'
import gc
import json
import threading
import time
import sys
import torch
from torch import nn
from transformers import AutoTokenizer, T5Config, T5EncoderModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import LABELS, remove_comments

DISPLAY = {
    'claude-3.5-haiku': 'Claude 3.5 Haiku', 'deepseek-chat': 'DeepSeek Chat',
    'gemini-2.5-flash-preview-05-20': 'Gemini 2.5 Flash', 'gpt-4.1': 'GPT-4.1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
}

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = T5EncoderModel(config)
        self.head = nn.Sequential(nn.Linear(config.d_model, 768), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(768, 5))

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])

class Engine:
    def __init__(self, device='cpu'):
        self.folder = ROOT / 'models'
        self.manifest = json.loads((self.folder / 'checkpoint_manifest.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.folder / 'tokenizer', local_files_only=True)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Start with --device cpu.')
        torch.set_num_threads(min(6, os.cpu_count() or 1))
        self.model = None
        self.condition = None
        self.lock = threading.Lock()

    def load(self, condition):
        if self.condition == condition:
            return
        if self.model is not None:
            self.model = None
            gc.collect()
            if self.device.type == 'cuda': torch.cuda.empty_cache()
        path = self.folder / f'codet5_remove_{condition}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {path.name}')
        config = T5Config.from_pretrained(self.folder / 'tokenizer', local_files_only=True)
        with torch.device('meta'):
            model = Classifier(config)
        weights = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
        model.load_state_dict(weights, strict=True, assign=True)
        self.model = model.to(self.device).eval()
        self.condition = condition

    def analyze(self, code, model='050', strip=False):
        if not isinstance(code, str) or not code.strip():
            raise ValueError('Paste some C code first.')
        if len(code) > 100000:
            raise ValueError('Use at most 100,000 characters.')
        if model not in ['050', '100', 'both']:
            raise ValueError('Choose a supported model.')
        if not isinstance(strip, bool):
            raise ValueError('The comment setting must be true or false.')
        with self.lock:
            started = time.perf_counter()
            source = remove_comments(code) if strip else code
            all_ids = self.tokenizer.encode('code: ' + source, truncation=False)
            inputs = self.tokenizer('code: ' + source, truncation=True, max_length=512,
                                    padding='longest', pad_to_multiple_of=8,
                                    return_tensors='pt', return_token_type_ids=False)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            results = []
            for condition in (['050', '100'] if model == 'both' else [model]):
                self.load(condition)
                with torch.inference_mode():
                    logits = self.model(**inputs)
                values = logits.float().softmax(-1)[0].cpu().tolist()
                ranking = sorted([{'label': label, 'name': DISPLAY[label], 'score': values[i]}
                                  for i, label in enumerate(LABELS)], key=lambda x: -x['score'])
                results.append({'condition': condition, 'prediction': ranking[0]['name'],
                                'label': ranking[0]['label'], 'scores': ranking})
            return {'results': results, 'input_tokens': len(all_ids), 'used_tokens': min(512, len(all_ids)),
                    'truncated': len(all_ids) > 512, 'comments_removed': strip,
                    'seconds': round(time.perf_counter() - started, 2), 'device': str(self.device),
                    'warning': 'Closed-set prediction among five LLMs. Scores are uncalibrated and do not establish authorship.'}

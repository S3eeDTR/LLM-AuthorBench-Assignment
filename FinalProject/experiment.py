"""Comment-removal augmentation study for the CSBP711 Course Project."""
from contextlib import nullcontext
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import io
import json
import random
import re
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, DataCollatorWithPadding, T5EncoderModel,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

WORK = Path('/content/course_project')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'Salesforce/codet5p-770m'
MODEL_REVISION = '3f7cc6e80aee6612f41253b5ee28f09f86faa35f'
SEED = 42
MAX_LENGTH = 512
MICRO_BATCH = 4
ACCUMULATION = 8
EPOCHS = 10
LEARNING_RATE = 5e-5
LABELS = sorted([
    'claude-3.5-haiku', 'deepseek-chat', 'gemini-2.5-flash-preview-05-20',
    'gpt-4.1', 'llama-3.3-70b-instruct',
])
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
DATA_HASH = 'e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c'
SPLIT_HASH = '8a3c8e2883844295fbddf5b78b96898478f606305972de4feb07a238254df5f4'
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*(?:\n|$)|/\*.*?\*/', re.S)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def require_gpu():
    assert torch.cuda.is_available(), 'Select an A100 GPU runtime first.'
    assert torch.cuda.is_bf16_supported(), 'This experiment requires bfloat16 support.'


def remove_comments(code):
    code = code.replace('\\\r\n', '').replace('\\\n', '')
    def replace(match):
        text = match.group()
        return ' ' + '\n' * text.count('\n') if text.startswith(('//', '/*')) else text
    return TOKEN.sub(replace, code)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_data():
    folder = WORK / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    inputs = [
        ('https://raw.githubusercontent.com/LLMauthorbench/LLMauthorbench/'
         '6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7/LLM-AuthorBench.json.zip',
         'LLM-AuthorBench.json.zip', DATA_HASH),
        ('https://raw.githubusercontent.com/S3eeDTR/LLM-AuthorBench-Assignment/'
         '2956a0ad07f49529026cacd33728018d6ebf96ca/data/splits.csv',
         'splits.csv', SPLIT_HASH),
    ]
    for url, name, expected in inputs:
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        content = path.read_bytes()
        if name.endswith('.csv'):
            content = content.replace(b'\r\n', b'\n')
        assert hashlib.sha256(content).hexdigest() == expected, f'File check failed: {name}'
        print('Verified:', name)


def load_data():
    with zipfile.ZipFile(WORK / 'data/LLM-AuthorBench.json.zip') as archive:
        data = pd.DataFrame(json.loads(archive.read('LLM-AuthorBench.json')))
    data = data.rename(columns={'c_code': 'code', 'model_name': 'label'})
    data['row_id'] = range(len(data))
    data = data[data.label.isin(LABELS)].copy()
    data['source_hash'] = data.code.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    splits = pd.read_csv(WORK / 'data/splits.csv')
    data = data.merge(splits[['row_id', 'label', 'source_hash', 'split', 'problem_id']],
                      on=['row_id', 'label', 'source_hash'], validate='one_to_one')
    assert len(data) == 20000 and not data.source_hash.duplicated().any()
    assert data.groupby('label').size().eq(4000).all()
    assert data.groupby('split').size().to_dict() == {'train': 13150, 'validation': 2978, 'test': 3872}
    for part in ['train', 'validation', 'test']:
        for column in ['problem_id', 'prompt', 'source_hash']:
            inside = set(data.loc[data.split == part, column])
            outside = set(data.loc[data.split != part, column])
            assert inside.isdisjoint(outside), f'Overlap in {column}'
    data['stripped_code'] = data.code.map(remove_comments)
    data['target'] = data.label.map(LABEL_TO_ID)
    return {part: data[data.split == part].reset_index(drop=True)
            for part in ['train', 'validation', 'test']}


def score_predictions(truth, prediction):
    return {
        'accuracy': accuracy_score(truth, prediction),
        'macro_f1': f1_score(truth, prediction, labels=LABELS, average='macro', zero_division=0),
        'weighted_f1': f1_score(truth, prediction, labels=LABELS, average='weighted', zero_division=0),
    }


def load_baselines(path, frames):
    required = [f'{model}_{variant}_predictions.csv'
                for model in ['svm', 'codet5'] for variant in ['original', 'stripped']]
    folder = WORK / 'reference'
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        metadata = json.loads(archive.read('baseline_metadata.json'))
        assert metadata['data_sha256'] == DATA_HASH
        assert metadata['normalized_split_sha256'] == SPLIT_HASH
        for name in required + ['baseline_metadata.json']:
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == manifest[name], f'Baseline hash failed: {name}'
            (folder / name).write_bytes(content)
    comparison = []
    for model in ['svm', 'codet5']:
        for variant in ['original', 'stripped']:
            table = pd.read_csv(folder / f'{model}_{variant}_predictions.csv')
            expected = frames['test'][['row_id', 'label']]
            joined = table.merge(expected, on='row_id', validate='one_to_one')
            assert len(table) == len(joined) == len(expected)
            assert joined.true.eq(joined.label).all()
            values = score_predictions(table.true, table.predicted)
            recorded = metadata[model]['metrics'][variant]
            assert all(abs(values[k] - recorded[k]) < 1e-10 for k in values)
            comparison.append({'model': metadata[model]['name'], 'variant': variant,
                               **values, **metadata[model]['cost']})
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return pd.DataFrame(comparison)


def prepare_tokens(frames):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    cache = {}
    for part, frame in frames.items():
        cache[part] = {}
        for variant, column in [('original', 'code'), ('stripped', 'stripped_code')]:
            texts = ['code: ' + code for code in frame[column]]
            cache[part][variant] = tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                                            padding=False, return_token_type_ids=False)
        print('Tokenized:', part, len(frame))
    return tokenizer, cache


def augmentation_mask(size, probability, epoch):
    assert probability in (0.0, 0.5, 1.0)
    mask = np.zeros(size, dtype=bool)
    count = int(round(size * probability))
    if count:
        rng = np.random.RandomState(SEED + epoch)
        mask[rng.choice(size, size=count, replace=False)] = True
    return mask


class CodeDataset(Dataset):
    def __init__(self, frame, tokens, stripped=False):
        self.frame = frame
        self.tokens = tokens
        self.mask = np.full(len(frame), stripped, dtype=bool)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        variant = 'stripped' if self.mask[index] else 'original'
        encoded = self.tokens[variant]
        return {'input_ids': encoded['input_ids'][index],
                'attention_mask': encoded['attention_mask'][index],
                'labels': int(self.frame.target.iloc[index])}


def make_loader(dataset, tokenizer, shuffle=False):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    return DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=shuffle,
                      collate_fn=collator, num_workers=0, pin_memory=DEVICE.type == 'cuda')


class CodeT5Authorship(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.head = nn.Sequential(nn.Linear(self.encoder.config.d_model, 768), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(768, len(LABELS)))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[:, 0])


def precision_context():
    return torch.autocast('cuda', dtype=torch.bfloat16) if DEVICE.type == 'cuda' else nullcontext()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    truth, prediction, probabilities = [], [], []
    for batch in loader:
        truth.extend(batch.pop('labels').tolist())
        inputs = {key: value.to(DEVICE) for key, value in batch.items()}
        with precision_context():
            logits = model(**inputs)
        prediction.extend(logits.argmax(-1).cpu().tolist())
        probabilities.extend(logits.float().softmax(-1).cpu().tolist())
    truth = [LABELS[i] for i in truth]
    prediction = [LABELS[i] for i in prediction]
    return score_predictions(truth, prediction), truth, prediction, probabilities


def run_name(probability):
    return f'remove_{int(probability * 100):03d}_seed{SEED}'


def run_training(probability, frames, tokens, tokenizer):
    require_gpu()
    folder = WORK / 'results' / run_name(probability)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'training_complete.json').exists():
        assert (folder / 'best_model.pt').exists(), 'Completed run is missing its checkpoint.'
        print('Training already completed:', folder.name)
        return folder
    if (folder / 'training_history.csv').exists():
        raise RuntimeError('A partial run exists. Save it before using a fresh output folder.')

    seed_everything()
    train_data = CodeDataset(frames['train'], tokens['train'])
    train_loader = make_loader(train_data, tokenizer, shuffle=True)
    validation = make_loader(CodeDataset(frames['validation'], tokens['validation']), tokenizer)
    model = CodeT5Authorship().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    steps = (len(train_loader) + ACCUMULATION - 1) // ACCUMULATION
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=steps * EPOCHS, num_cycles=1)
    settings = {
        'status': 'training', 'comment_removal_probability': probability,
        'model': MODEL_NAME, 'model_revision': MODEL_REVISION, 'seed': SEED,
        'max_length': MAX_LENGTH, 'micro_batch': MICRO_BATCH, 'accumulation': ACCUMULATION,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': 0.01,
        'warmup_steps': 200, 'optimizer': 'AdamW', 'scheduler': 'cosine restarts; one cycle',
        'selection': 'original-code validation accuracy; earliest epoch wins ties',
        'prefix': 'code: ', 'precision': 'fp32 parameters + bf16 autocast',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU smoke test',
        'versions': {name: importlib.metadata.version(name)
                     for name in ['torch', 'transformers', 'numpy', 'pandas', 'scikit-learn']},
        'data_sha256': DATA_HASH, 'normalized_split_sha256': SPLIT_HASH,
        'augmentation_rng': 'NumPy RandomState(seed + epoch), independent of torch RNG',
    }
    (folder / 'settings.json').write_text(json.dumps(settings, indent=2))
    changed = frames['train'].code.ne(frames['train'].stripped_code).to_numpy()
    history, best_accuracy, best_epoch = [], -1.0, None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_data.mask = augmentation_mask(len(train_data), probability, epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, epoch_start = [], time.perf_counter()
        progress = tqdm(train_loader, desc=f'{int(probability * 100)}% removal | epoch {epoch}/{EPOCHS}')
        for step, batch in enumerate(progress):
            target = batch.pop('labels').to(DEVICE)
            inputs = {key: value.to(DEVICE) for key, value in batch.items()}
            group_start = (step // ACCUMULATION) * ACCUMULATION
            group_size = min(ACCUMULATION, len(train_loader) - group_start)
            with precision_context():
                loss = nn.functional.cross_entropy(model(**inputs), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss. Stop and inspect this run.')
            (loss / group_size).backward()
            losses.append(loss.item())
            if (step + 1) % ACCUMULATION == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                progress.set_postfix(loss=f'{np.mean(losses[-100:]):.4f}')
        validation_metrics, _, _, _ = predict(model, validation)
        record = {
            'epoch': epoch, 'training_loss': float(np.mean(losses)),
            'validation_accuracy': validation_metrics['accuracy'],
            'validation_macro_f1': validation_metrics['macro_f1'],
            'requested_stripped_programs': int(train_data.mask.sum()),
            'actually_changed_programs': int((train_data.mask & changed).sum()),
            'epoch_seconds': time.perf_counter() - epoch_start,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
        if validation_metrics['accuracy'] > best_accuracy:
            best_accuracy, best_epoch = validation_metrics['accuracy'], epoch
            torch.save(model.state_dict(), folder / 'best_model.pt')
        print(json.dumps(record), flush=True)
    complete = {
        'status': 'training_completed', 'best_epoch': best_epoch,
        'best_validation_accuracy': best_accuracy,
        'training_validation_checkpoint_seconds': time.perf_counter() - started,
        'checkpoint_bytes': (folder / 'best_model.pt').stat().st_size,
        'parameter_count': settings['parameter_count'],
    }
    (folder / 'training_complete.json').write_text(json.dumps(complete, indent=2))
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Saved:', folder.name, '| best epoch:', best_epoch)
    return folder


def evaluate_run(probability, frames, tokens, tokenizer):
    folder = WORK / 'results' / run_name(probability)
    assert (folder / 'training_complete.json').exists(), f'Training is unfinished: {folder.name}'
    if (folder / 'metrics.json').exists():
        return json.loads((folder / 'metrics.json').read_text())
    model = CodeT5Authorship().to(DEVICE)
    model.load_state_dict(torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=True))
    result = {'status': 'completed', 'comment_removal_probability': probability}
    for variant in ['original', 'stripped']:
        dataset = CodeDataset(frames['test'], tokens['test'], stripped=variant == 'stripped')
        values, truth, prediction, probabilities = predict(model, make_loader(dataset, tokenizer))
        result[variant] = values
        output = pd.DataFrame({'row_id': frames['test'].row_id, 'true': truth, 'predicted': prediction})
        for i, label in enumerate(LABELS):
            output['prob_' + label] = [p[i] for p in probabilities]
        output.to_csv(folder / f'{variant}_predictions.csv', index=False)
        matrix = confusion_matrix(truth, prediction, labels=LABELS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(folder / f'{variant}_confusion.csv')
        report = classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(folder / f'{variant}_class_scores.csv')
    result['accuracy_drop_pp'] = 100 * (result['original']['accuracy'] - result['stripped']['accuracy'])
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_both(frames, tokens, tokenizer):
    for probability in [0.5, 1.0]:
        folder = WORK / 'results' / run_name(probability)
        assert (folder / 'training_complete.json').exists(), f'Finish training first: {folder.name}'
    return {run_name(p): evaluate_run(p, frames, tokens, tokenizer) for p in [0.5, 1.0]}


def comparison_table(baselines):
    rows = baselines.to_dict('records')
    for probability in [0.5, 1.0]:
        folder = WORK / 'results' / run_name(probability)
        result = json.loads((folder / 'metrics.json').read_text())
        cost = json.loads((folder / 'training_complete.json').read_text())
        for variant in ['original', 'stripped']:
            rows.append({'model': f'CodeT5 {int(probability * 100)}% removal', 'variant': variant,
                         **result[variant], 'training_seconds': cost['training_validation_checkpoint_seconds'],
                         'timing_scope': 'GPU training + validation + checkpoint writing',
                         'parameter_count': cost['parameter_count'], 'size_MB': cost['checkpoint_bytes'] / 1e6})
    table = pd.DataFrame(rows)
    table.to_csv(WORK / 'results/comparison.csv', index=False)
    reference = baselines[(baselines.model == 'CodeT5 0% removal') & (baselines.variant == 'original')].iloc[0]
    decisions = []
    for probability in [0.5, 1.0]:
        model_name = f'CodeT5 {int(probability * 100)}% removal'
        clean = table[(table.model == model_name) & (table.variant == 'original')].iloc[0]
        stripped = table[(table.model == model_name) & (table.variant == 'stripped')].iloc[0]
        old_stripped = baselines[(baselines.model == 'CodeT5 0% removal') & (baselines.variant == 'stripped')].iloc[0]
        decisions.append({'model': model_name,
                          'stripped_macro_f1_gain': stripped.macro_f1 - old_stripped.macro_f1,
                          'clean_accuracy_change_pp': 100 * (clean.accuracy - reference.accuracy),
                          'clean_accuracy_within_2pp': bool(clean.accuracy >= reference.accuracy - 0.02)})
    pd.DataFrame(decisions).to_csv(WORK / 'results/tradeoffs.csv', index=False)
    return table, pd.DataFrame(decisions)


def save_figures(table):
    import matplotlib.pyplot as plt
    models = ['Assignment 1 SVM', 'CodeT5 0% removal', 'CodeT5 50% removal', 'CodeT5 100% removal']
    labels = ['SVM', 'CodeT5 0%', 'CodeT5 50%', 'CodeT5 100%']
    for metric, title in [('accuracy', 'Accuracy (%)'), ('macro_f1', 'Macro-F1')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(models))
        for shift, variant, color in [(-0.18, 'original', '#147D85'), (0.18, 'stripped', '#F2A541')]:
            values = [table[(table.model == model) & (table.variant == variant)][metric].iloc[0] for model in models]
            if metric == 'accuracy':
                values = [100 * value for value in values]
            bars = ax.bar(x + shift, values, width=0.36, label=variant, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylabel(title)
        ax.set_ylim(0, 105 if metric == 'accuracy' else 1.05)
        ax.set_title('Original and comment-stripped test code')
        ax.legend()
        fig.tight_layout()
        fig.savefig(WORK / 'results' / f'{metric}_comparison.png', dpi=180)
        plt.close(fig)


def evidence_bundle(name='course_project_evidence.zip'):
    target = WORK / name
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder_name in ['results', 'reference']:
            folder = WORK / folder_name
            if folder.exists():
                for path in folder.rglob('*'):
                    if path.is_file() and path.suffix != '.pt':
                        archive.write(path, path.relative_to(WORK))
    return target

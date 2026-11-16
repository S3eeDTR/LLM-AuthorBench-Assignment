"""Optional GPU runner. The Colab notebook is the recommended entry point."""
import os
os.environ['USE_TF'] = '0'
import argparse
from pathlib import Path
import experiment as exp

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--all', action='store_true', help='Train both runs, then evaluate and export.')
args = parser.parse_args()
if not args.all:
    parser.error('Use --all to start both full training runs. See README.txt first.')
exp.WORK = args.workdir
exp.require_gpu()
exp.download_data()
frames = exp.load_data()
baselines = exp.load_baselines(exp.WORK / 'baseline_reference.zip', frames)
tokenizer, tokens = exp.prepare_tokens(frames)
for probability in [0.5, 1.0]:
    exp.run_training(probability, frames, tokens, tokenizer)
for probability in [0.5, 1.0]:
    exp.evaluate_run(probability, frames, tokens, tokenizer)
table, tradeoffs = exp.comparison_table(baselines)
exp.save_figures(table)
print(table.to_string(index=False))
print(tradeoffs.to_string(index=False))
print('Saved evidence:', exp.evidence_bundle())

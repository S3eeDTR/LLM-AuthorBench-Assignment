const $ = id => document.getElementById(id);
const transfer = sessionStorage.getItem('generated-code-transfer');
sessionStorage.removeItem('generated-code-transfer');
const example = `#include <stdio.h>\n\n/* Sum the integers from 1 to n. */\nint main(void) {\n    int n;\n    if (scanf("%d", &n) != 1 || n < 0) return 1;\n    long long sum = 0;\n    for (int i = 1; i <= n; ++i) {\n        sum += i;\n    }\n    printf("%lld\\n", sum);\n    return 0;\n}\n`;
const count = () => { $('count').textContent = `${$('code').value.length.toLocaleString()} characters`; };
let busy = false;
let samples = [];
function resetResults() { $('results').hidden = true; $('empty').hidden = false; $('status').className = 'status'; }
function customInput() { $('sample').value = ''; $('sample-note').textContent = 'Custom input. Select a benchmark example above to load its original code.'; resetResults(); count(); }
$('code').addEventListener('input', () => { customInput(); $('status').textContent = 'Input changed. Click Analyze code for a new prediction.'; });
$('example').onclick = () => { $('code').value = example; customInput(); $('status').textContent = 'Illustrative example with no known generating LLM. Results are not an accuracy test.'; };
$('clear').onclick = () => { $('code').value = ''; customInput(); $('status').textContent = 'Runs locally. Your code is never executed or saved.'; };
$('sample').onchange = () => {
    const selected = samples.find(item => String(item.row_id) === $('sample').value);
    if (!selected) { $('sample-note').textContent = 'Custom input. Paste code or choose a benchmark sample.'; return; }
    $('code').value = selected.code; count(); resetResults();
    $('sample-note').textContent = `Recorded generator: ${selected.name} · Test row ${selected.row_id}. Original code, not selected for a correct prediction.`;
    $('status').textContent = 'Benchmark sample loaded. Choose your model and comment setting, then analyze.';
};
$('file').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 400000) { $('status').textContent = 'Choose a smaller file (at most 100,000 characters).'; return; }
    const text = await file.text();
    if (text.length > 100000) { $('status').textContent = 'Use at most 100,000 characters.'; return; }
    $('code').value = text; customInput(); $('status').textContent = `Loaded ${file.name}. Ready to analyze.`;
};
function node(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function render(data) {
    const container = $('results'); container.replaceChildren();
    for (const result of data.results) {
        const block = node('section', 'prediction');
        block.append(node('span', 'result-label', `${result.condition === '050' ? '50%' : '100%'} removal model · closest match`));
        block.append(node('h3', '', result.prediction));
        const ranks = node('div');
        for (const score of result.scores) {
            const row = node('div', 'score-row');
            // CSP permits external styles only; use a native progress element for the score.
            const progress = document.createElement('progress'); progress.max = 1; progress.value = score.score;
            progress.setAttribute('aria-label', `${score.name} model score`); progress.className = 'score-progress';
            row.append(node('span', '', score.name), progress, node('span', 'score-value', `${(100 * score.score).toFixed(1)}%`)); ranks.append(row);
        }
        block.append(ranks);
        if(data.analysis_mode==='chunks'){
            const details=node('details');details.append(node('summary','',`View ${result.chunks.length} chunk predictions`));
            details.append(node('p','result-meta','Code token ranges exclude the repeated prefix and special tokens.'));
            for(const chunk of result.chunks)details.append(node('p','result-meta',`Chunk ${chunk.chunk} · code tokens ${chunk.code_token_start}–${chunk.code_token_end} · ${chunk.model_tokens} model tokens · ${chunk.prediction} (${(chunk.score*100).toFixed(1)}%)`));
            block.append(details);
        }
        container.append(block);
    }
    const meta = node('p', 'result-meta', `${data.used_tokens} of ${data.input_tokens} tokens used · ${data.comments_removed ? 'Comments removed' : 'Comments kept'} · ${data.seconds}s · ${data.device.toUpperCase()}. Scores are uncalibrated.`);
    container.append(meta);
    if(data.analysis_mode==='chunks')container.append(node('p','result-meta',`Full-code mode: ${data.chunk_count} window(s), ${data.overlap_tokens}-token overlap. Equal-weight average of chunk scores. This optional method has not been evaluated in the report; accuracy is not guaranteed to improve.`));
    if (data.truncated) container.append(node('p', 'result-meta truncation', 'Input exceeds the 512-token limit. Only the beginning was analyzed.'));
    $('empty').hidden = true; container.hidden = false;
}
async function analyze() {
    if (busy) return;
    if (!$('code').value.trim()) { $('status').textContent = 'Paste C code or try the example first.'; $('code').focus(); return; }
    busy = true; $('analyze').disabled = true; $('analyze').textContent = 'Analyzing…'; $('status').className = 'status';
    $('status').textContent = 'Loading the checkpoint and analyzing code. The first request takes longer.';
    const submitted = { code: $('code').value, model: $('model').value, strip: $('preprocess').value === 'strip', mode:$('coverage').value };
    try {
        const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(submitted) });
        const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Analysis failed.');
        if (submitted.code !== $('code').value || submitted.model !== $('model').value || submitted.strip !== ($('preprocess').value === 'strip') || submitted.mode!==$('coverage').value) {
            resetResults(); $('status').textContent = 'Input or settings changed. Click Analyze code again.'; return;
        }
        render(data); $('status').textContent = 'Analysis complete. No code was executed or saved.';
    } catch (error) { $('status').className = 'status error'; $('status').textContent = error.message; }
    finally { busy = false; $('analyze').disabled = false; $('analyze').textContent = 'Analyze code ↗'; }
}
$('analyze').onclick = analyze;
for (const id of ['model', 'preprocess','coverage']) $(id).addEventListener('change', () => { resetResults(); $('status').textContent = 'Settings changed. Click Analyze code for a new prediction.'; });
$('code').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyze(); });
fetch('/api/status').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(data => { $('connection').textContent = `Local model · ${data.device.toUpperCase()}`; }).catch(() => { $('connection').textContent = 'Server unavailable'; $('status').textContent = 'Start the local server, then reload this page.'; });
fetch('/api/samples').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(data => {
    samples = data.samples;
    $('sample').replaceChildren();
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a model’s test sample…'; $('sample').append(placeholder);
    for (const sample of samples) {
        const option = document.createElement('option'); option.value = String(sample.row_id); option.textContent = `${sample.name} · row ${sample.row_id}`; $('sample').append(option);
    }
    $('sample').disabled = false;
}).catch(() => { $('sample').options[0].textContent = 'Samples unavailable'; $('sample-note').textContent = 'Paste code or use Open .c file instead.'; });
if (transfer) {
    try { const sample=JSON.parse(transfer); $('code').value=sample.code; customInput(); $('sample-note').textContent='Fresh generated sample · returned model: '+sample.model; $('status').textContent='Generated code loaded. Click Analyze code to test the classifier.'; } catch (_) {}
}

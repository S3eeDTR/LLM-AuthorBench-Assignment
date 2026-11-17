const $=id=>document.getElementById(id);
let models=[], records=[];
let comparisons=[], testing=false;
function element(tag,text,cls){let e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=element('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function codeOnly(text){const match=text.trim().match(/^```(?:c|C)?\s*\n([\s\S]*?)\n```$/);return match?match[1]:text;}
$('forget').onclick=()=>{$('api-key').value='';};
fetch('/api/generator-models').then(async response=>{const data=await response.json();if(!response.ok)throw Error(data.error);return data;}).then(data=>{
 models=data.models;$('models').replaceChildren();
 for(const model of models){const row=element('label',undefined,'generator-model');const check=document.createElement('input');check.type='checkbox';check.value=model.id;check.disabled=!model.available;check.checked=model.available;
 const details=element('span');details.append(element('strong',model.name),element('small',model.id+' · '+(model.available?'Listed as available':'Unavailable — skipped')));row.append(check,details);$('models').append(row);}
 $('generate').disabled=!models.some(m=>m.available);
}).catch(error=>{$('models').textContent=error.message;});
$('download-all').onclick=()=>download('generation_results.json',JSON.stringify(records,null,2),'application/json');
$('generate').onclick=async()=>{
 if(testing)return;
 const key=$('api-key').value.trim(),prompt=$('prompt').value.trim();
 const selected=Array.from($('models').querySelectorAll('input:checked')).map(e=>e.value);
 if(!key||!prompt||!selected.length){$('gen-status').textContent='Enter your key, a prompt and select at least one available model.';return;}
 $('generate').disabled=true;$('prompt').disabled=true;$('download-all').disabled=true;
 $('models').querySelectorAll('input').forEach(e=>e.disabled=true);
 records=[];comparisons=[];$('outputs').replaceChildren();clearComparison();$('test-all').disabled=true;$('import-results').disabled=true;$('test-benchmark').disabled=true;
 try{for(const [index,id] of selected.entries()){
  $('gen-status').textContent=`Generating ${index+1} of ${selected.length}…`;
  const card=element('section',undefined,'panel generated-card');card.append(element('h2',models.find(m=>m.id===id).name));const message=element('p','Waiting for OpenRouter…','small');card.append(message);$('outputs').append(card);
  try{const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key:key,prompt,model:id})});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'Generation failed.');
   records.push({...data,generated_at:new Date().toISOString()});message.textContent=`Returned: ${data.returned_model||'not reported'} · Provider: ${data.provider||'not reported'} · Finish: ${data.finish_reason||'not reported'}`;
   const output=element('textarea');output.readOnly=true;output.value=data.code;output.setAttribute('aria-label','Generated code from '+id);card.append(output);
   if(data.finish_reason==='length')card.append(element('p','Output hit the token limit and may be incomplete.','status error'));
   const bar=element('div',undefined,'generation-actions');const save=element('button','Download .c','text-button');save.onclick=()=>download(id.replace(/[^a-z0-9-]/gi,'_')+'.c',codeOnly(data.code),'text/plain');
   const analyze=element('button','Use in classifier','text-button');analyze.onclick=()=>{sessionStorage.setItem('generated-code-transfer',JSON.stringify({code:codeOnly(data.code),model:data.returned_model||id}));window.location.href='/';};bar.append(save,analyze);card.append(bar);
  }catch(error){message.textContent=error.message;records.push({requested_model:id,prompt,error:error.message,generated_at:new Date().toISOString()});}
 }}finally{$('generate').disabled=false;$('prompt').disabled=false;$('download-all').disabled=!records.length;$('models').querySelectorAll('input').forEach(e=>e.disabled=!models.find(m=>m.id===e.value).available);$('gen-status').textContent='Finished. Downloads exclude your API key. Errors are preserved in the JSON results.'; $('import-results').disabled=false;$('test-benchmark').disabled=false;$('test-all').disabled=!records.some(r=>r.code&&!r.error);}
 if(records.some(r=>r.code&&!r.error))await testAll();
};

const labelMap={
 'anthropic/claude-3.5-haiku':'claude-3.5-haiku',
 'deepseek/deepseek-chat':'deepseek-chat',
 'google/gemini-2.5-flash-preview-05-20':'gemini-2.5-flash-preview-05-20',
 'google/gemini-2.5-flash':'gemini-2.5-flash-preview-05-20',
 'openai/gpt-4.1':'gpt-4.1',
 'meta-llama/llama-3.3-70b-instruct':'llama-3.3-70b-instruct'
};
function clearComparison(){comparisons=[];$('comparison-body').replaceChildren();$('batch-summary').replaceChildren();$('comparison').hidden=true;$('download-comparison').disabled=true;}
function addComparison(row){
 comparisons.push(row);const tr=element('tr');
 for(const value of [row.actual,row.classifier,row.predicted||'Failed',row.match,row.score===undefined?'—':(100*row.score).toFixed(1)+'%',row.gpt_score===undefined?'—':(100*row.gpt_score).toFixed(1)+'%',row.used_tokens===undefined?'—':`${row.used_tokens}/${row.input_tokens}`,row.notes])tr.append(element('td',String(value||'')));
 $('comparison-body').append(tr);$('comparison').hidden=false;$('download-comparison').disabled=false;
 $('batch-summary').replaceChildren();
 for(const condition of [...new Set(comparisons.map(row=>row.condition))]){
  const rows=comparisons.filter(row=>row.condition===condition),successful=rows.filter(row=>!row.error),gpt=successful.filter(row=>row.predicted_label==='gpt-4.1');
  const frequencies={};for(const row of successful)frequencies[row.predicted]=(frequencies[row.predicted]||0)+1;
  const assessed=successful.filter(row=>row.exact_reference),matches=assessed.filter(row=>row.match==='Match');
  $('batch-summary').append(element('strong',`${Number(condition)}% removal model`),element('p',`${gpt.length} of ${successful.length} analyzed samples predicted as GPT-4.1. ${rows.filter(row=>row.error).length} analysis failures.`),element('p',`Prediction counts: ${Object.entries(frequencies).map(([name,count])=>name+': '+count).join(' · ')||'None yet'}`),element('p',`Label matches on comparable recorded IDs: ${matches.length}/${assessed.length}. Version-shift and unknown IDs excluded. This small exploratory batch is not a benchmark accuracy estimate.`));
 }
}
async function testAll(){
 if(testing)return;const batch=records.filter(r=>typeof r.code==='string'&&r.code.trim()&&!r.error);if(!batch.length)return;
 const strip=$('batch-comments').value==='strip';
 const mode=$('batch-coverage').value;
 const conditions=$('batch-model').value==='both'?['050','100']:[$('batch-model').value];
 testing=true;clearComparison();$('comparison-caption').textContent=`Local inference · ${conditions.map(c=>Number(c)+'%').join(' and ')} model(s) · comments ${strip?'removed':'kept'} · ${mode==='chunks'?'full code using chunks':'first 512 tokens'} · uncalibrated scores`;$('test-all').disabled=true;$('generate').disabled=true;$('import-results').disabled=true;$('batch-comments').disabled=true;$('batch-model').disabled=true;$('batch-coverage').disabled=true;$('test-benchmark').disabled=true;
 try{for(const condition of conditions){for(const [index,record] of batch.entries()){
  $('batch-status').textContent=`Testing ${index+1} of ${batch.length} with the ${Number(condition)}% model…`;
  const actual=record.source==='heldout_benchmark'?`${record.actual_label} (test row ${record.sample_row})`:record.returned_model||record.requested_model||'Unknown generator';
  const expected=labelMap[record.requested_model];
  const compatible=record.returned_model===record.requested_model||(record.requested_model==='openai/gpt-4.1'&&/^openai\/gpt-4\.1-\d{4}-\d{2}-\d{2}$/.test(record.returned_model||''))||(record.requested_model==='deepseek/deepseek-chat'&&record.returned_model==='deepseek/deepseek-chat-v3');
  const exact=Boolean(expected&&((record.source==='heldout_benchmark'&&record.actual_label===expected)||(compatible&&record.requested_model!=='google/gemini-2.5-flash')));
  try{const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:codeOnly(record.code),model:condition,strip,mode})});const data=await response.json();if(!response.ok)throw Error(data.error||'Analysis failed.');
   const result=data.results[0],match=result.label===expected;
   const notes=[];if(mode==='chunks')notes.push(data.chunk_count+' chunks; averaged scores');if(data.truncated)notes.push('Truncated at 512 tokens');if(record.finish_reason==='length')notes.push('Generation hit output limit');if(!exact)notes.push('Different version / unverified ID');
   const row={actual,condition,analysis_mode:mode,chunk_count:data.chunk_count,chunks:result.chunks,classifier:Number(condition)+'% removal',requested_model:record.requested_model,provider:record.provider||'',predicted:result.prediction,predicted_label:result.label,match:exact?(match?'Match':'Mismatch'):(expected?(match?'Family match only':'Different label'):'Not comparable'),exact_reference:exact,score:result.scores[0].score,gpt_score:result.scores.find(s=>s.label==='gpt-4.1').score,input_tokens:data.input_tokens,used_tokens:data.used_tokens,truncated:data.truncated,comments_removed:strip,notes:notes.join('; '),scores:result.scores};
   record[`analysis_${Number(condition)}_${strip?'stripped':'original'}_${mode}`]=row;addComparison(row);
  }catch(error){const row={actual,condition,classifier:Number(condition)+'% removal',requested_model:record.requested_model,match:'Not assessed',notes:error.message,error:true};record[`analysis_${Number(condition)}_${strip?'stripped':'original'}_${mode}`]=row;addComparison(row);}
 }}}finally{testing=false;$('test-all').disabled=false;$('generate').disabled=!models.some(m=>m.available);$('import-results').disabled=false;$('batch-comments').disabled=false;$('batch-model').disabled=false;$('batch-coverage').disabled=false;$('test-benchmark').disabled=false;$('batch-status').textContent='Batch testing complete. Counts, scores and truncation help investigate the pattern; they do not prove why it occurs.';}
}
$('test-all').onclick=testAll;
$('batch-comments').onchange=()=>{clearComparison();$('batch-status').textContent='Comment setting changed. Click Test all to compare the new condition.';};
$('batch-coverage').onchange=()=>{clearComparison();$('batch-status').textContent='Coverage changed. Click Test all to rerun the saved code.';};
$('batch-model').onchange=()=>{clearComparison();$('batch-status').textContent='Classifier changed. Click Test all to rerun the same samples.';};
$('import-results').onchange=async event=>{
 const file=event.target.files[0];if(!file)return;
 try{if(file.size>10000000)throw Error('Choose a JSON file smaller than 10 MB.');const parsed=JSON.parse(await file.text());if(!Array.isArray(parsed)||!parsed.length||parsed.length>100)throw Error('Expected a JSON array with 1–100 generation records.');
 const allowed=['code','requested_model','returned_model','provider','response_id','finish_reason','usage','prompt','temperature','max_tokens','generated_at','error','source','sample_row','actual_label'];
 records=parsed.map(source=>{if(!source||typeof source!=='object')throw Error('Invalid generation record.');let row={};for(const key of allowed)if(Object.hasOwn(source,key))row[key]=source[key];if(row.code&&(typeof row.code!=='string'||row.code.length>100000))throw Error('A code sample exceeds 100,000 characters.');return row;});
 clearComparison();$('outputs').replaceChildren();$('download-all').disabled=false;$('test-all').disabled=!records.some(r=>r.code&&!r.error);$('gen-status').textContent=`Imported ${records.length} records. No generation requests or API credits used.`;await testAll();
 }catch(error){$('batch-status').textContent=error.message;}
};
$('test-benchmark').onclick=async()=>{
 if(testing)return;$('test-benchmark').disabled=true;
 try{const response=await fetch('/api/samples');const data=await response.json();if(!response.ok)throw Error(data.error||'Could not load benchmark examples.');
 records=data.samples.map(sample=>({code:sample.code,requested_model:Object.keys(labelMap).find(id=>labelMap[id]===sample.label),actual_label:sample.label,sample_row:sample.row_id,source:'heldout_benchmark',provider:'Dataset metadata'}));
 $('outputs').replaceChildren(element('p','Loaded five unchanged held-out benchmark programs, one per generator. These were unseen during training but have already been evaluated. No examples were selected based on prediction correctness.'));
 $('download-all').disabled=false;$('gen-status').textContent='Using held-out benchmark examples. No API key or credits used.';await testAll();
 }catch(error){$('batch-status').textContent=error.message;}finally{$('test-benchmark').disabled=false;}
};
$('download-comparison').onclick=()=>{
 const keys=['actual','classifier','condition','requested_model','provider','predicted','match','score','gpt_score','input_tokens','used_tokens','truncated','comments_removed','analysis_mode','chunk_count','notes'];
 const cell=value=>'"'+String(value??'').replace(/"/g,'""')+'"';
 download('actual_vs_predicted.csv',[keys.map(cell).join(','),...comparisons.map(row=>keys.map(key=>cell(row[key])).join(','))].join('\r\n'),'text/csv');
};

fetch("/api/prompts").then(r=>r.json()).then(data=>{for(const [i,prompt] of data.prompts.entries()){const option=element("option",`${i+1}. ${prompt.task}`);option.value=prompt.task;$("prompt-preset").append(option);}$("prompt-preset").onchange=()=>{const chosen=data.prompts.find(p=>p.task===$("prompt-preset").value);if(chosen)$("prompt").value=chosen.prompt;};$("prompt").addEventListener("input",()=>{$("prompt-preset").value="";});}).catch(()=>{});

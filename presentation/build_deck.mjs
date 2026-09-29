import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const root=process.env.PROJECT_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const skill=process.env.SKILL_DIR;
if(!skill || !process.env.RUNTIME_PYTHON) throw new Error('Set SKILL_DIR and RUNTIME_PYTHON for the presentation runtime.');
const { finalizePresentation,resolvePresentationFont }=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const build=path.join(root,'tmp/deck'); const out=path.join(root,'presentation');
await fs.mkdir(build,{recursive:true}); await fs.mkdir(out,{recursive:true});
const d=JSON.parse(await fs.readFile(path.join(root,'results/presentation_data.json'),'utf8'));
const font=resolvePresentationFont({fontFamily:'Arial'});
const pres=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#152E38', teal='#117C83', muted='#50636B', paper='#F7FAFA';
function text(s,value,x,y,w,h,size=28,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value; t.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none',wrap:'word',insets:{left:0,right:0,top:0,bottom:0}}; return t;
}
function slide(title){const s=pres.slides.add();s.background.fill=paper;text(s,title,60,42,1160,80,44,ink,true);return s;}
function notes(s,value){s.speakerNotes.textFrame.setText(value);}
function table(s,values,x,y,width,height,colWidths,fontSize=25,highlight=-1){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:colWidths,values});
 t.borders.assign({fill:'#D6E1E4',width:1,style:'solid'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=height/values.length;
  for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?ink:r===highlight?'#E0F0ED':'#FFFFFF';
   cell.text.style={typeface:font,fontSize:r===0?fontSize-1:fontSize,color:r===0?'#FFFFFF':ink,bold:r===0||r===highlight,alignment:c===0?'left':'center',autoFit:'none'};
  }
 }
 return t;
}
const repo='https://github.com/S3eeDTR/LLM-AuthorBench-Assignment';
const source='https://github.com/LLMauthorbench/LLMauthorbench';
const citation='Bisztray et al., I Know Which LLM Wrote Your Code Last Summer, https://arxiv.org/abs/2506.17323 . Collection description only; all reported metrics come from local experiments.';
let s=pres.slides.add();s.background.fill=ink;
text(s,'LLM-Generated Code\nAuthorship Attribution',60,70,1130,160,58,'#FFFFFF',true);
text(s,'Can standard machine learning identify the LLM behind a C program?',60,256,1070,90,32,'#E4F0F2');
text(s,'32,000 C programs     8 LLMs     4,000 programs per class',60,376,1140,52,32,'#6DDBCA',true);
text(s,'LLM-AuthorBench connects this source-code experiment to our research on attribution of LLM-generated code.',60,462,1110,74,28,'#FFFFFF');
const link=text(s,repo,60,574,1160,35,23,'#6DDBCA');
link.text=[[{run:repo,link:{uri:repo,isExternal:true}}]];
text(s,'Dataset: github.com/LLMauthorbench/LLMauthorbench',60,624,1150,33,23,'#CCDADF');
notes(s,citation+'\nThe authors describe generation from 300 parameterized task templates, with duplicate removal and compilation filtering before release. We did not compile or execute programs. Our inspected release has 10,262 exact prompt strings and eight balanced labels. Licence permission is unverified and must be resolved before submission. Group names and contributions are pending user completion. Assignment: CSBP711 Fall 2026, due October 4.');
s=slide('Problem families stay in one split');
text(s,'259 inferred families group prompt parameters and audited task variants',60,144,1160,44,28,teal,true);
const splitRows=[['Split','Families','Programs'],...d.splits.map(r=>[r.split[0].toUpperCase()+r.split.slice(1),String(r.problems),r.samples.toLocaleString('en-US')])];
table(s,splitRows,60,215,570,260,[245,150,175],26);
text(s,'50,000 character TF-IDF features\nFit on training code only',688,216,520,91,28,ink,true);
text(s,'0 empty or malformed records\n0 exact source duplicates\nComments and identifiers retained',688,335,520,130,27);
text(s,'Disjoint groups, exact prompts and code hashes pass assertions.\nThe dataset has no authoritative task IDs, so semantic overlap remains a limitation.',60,518,1155,100,27);
text(s,'Research-use licence and instructor dataset claim remain unconfirmed.',60,641,1150,33,23,'#825A20');
notes(s,'Sources: results/dataset_statistics.json; split_summary.csv; split_metadata.json; upstream_template_audit.json. Seed 42. Split targets are 70/15/15 percent of families, not samples; actual sample shares differ. The independent audit matched 298 of 300 upstream templates and found no matched template spanning multiple inferred groups. Grouping is conservative and can merge related distinct tasks. Exact semantic separation cannot be certified. Case is preserved; sklearn internally collapses repeated whitespace. No JSON metadata enters features. Licence evidence: results/license_check.json and docs/DATASET_PERMISSION.md.');
s=slide('Four algorithms share the same evidence');
text(s,'C source text with comments and identifiers',60,145,1160,45,30,teal,true);
text(s,'Character TF-IDF with 3–5 grams and sublinear term frequency',60,204,1160,47,29);
const alg=[['Naive Bayes','Simple probabilistic baseline'],['Logistic Regression','Regularized linear comparison'],['Linear SVM','Margin-based linear comparison'],['Random Forest','Nonlinear tree ensemble comparison']];
for(let i=0;i<alg.length;i++){text(s,alg[i][0],60,298+i*61,390,48,29,ink,true);text(s,alg[i][1],490,298+i*61,710,48,27);}
text(s,'Same split and matrices. Seed 42 where supported. No tuning search.\nSelect by validation Macro-F1, then report held-out test performance.',60,572,1150,85,27);
notes(s,'Sources: src/features.py and src/train_models.py. MultinomialNB alpha=1. Logistic Regression C=1, lbfgs, max_iter=3000. LinearSVC C=1, dual=auto, max_iter=10000. Forest 150 trees, max_depth=40, min_samples_leaf=2, max_features=sqrt, n_jobs=4. Macro-F1 weights each class equally. Uniform random expected accuracy 12.5%. Majority baseline uses the training majority class. Library reference: https://scikit-learn.org/stable/ .');
s=slide('Linear SVM leads on the held-out test set');
text(s,'Test metrics after selection using original validation Macro-F1',60,143,1160,44,28,teal,true);
const original=d.original.filter(r=>r.split==='test');
const metricRows=[['Model','Accuracy','Macro-F1','Fit (s)','Size (MB)'],...original.map(r=>[r.model,(r.accuracy*100).toFixed(2)+'%',r.macro_f1.toFixed(4),r.training_seconds.toFixed(2),r.model_MB.toFixed(3)])];
table(s,metricRows,60,217,1160,300,[380,190,190,190,210],26,3);
text(s,'SVM validation Macro-F1: 0.6182\nUniform chance accuracy: 12.5%',60,559,550,86,28);
text(s,'Single-run classifier time and compressed size.\nShared TF-IDF size and fit time reported separately.',654,559,560,86,25,muted);
notes(s,'Sources: results/model_results.csv and original_features.json. Original TF-IDF extraction 25.8312 seconds and vectorizer 0.47146 MB. Classifier timings exclude feature extraction and serialization. Metrics use the same 6,108 test samples. Validation uses 4,777 samples. SVM remains the original validation-selected winner; the ablation does not reselect the original winner. Largest SVM directional confusion: Qwen to Llama, 171 programs. Single split and no confidence interval; do not imply small score gaps are statistically established.');
s=slide('Comments help, but SVM stays first');
text(s,'One ablation: remove comments and refit shared TF-IDF on training code',60,140,1160,43,27,teal,true);
const ab=d.ablation.filter(r=>r.split==='validation').sort((a,b)=>a.rank_original-b.rank_original);
const vals=[['Validation F1','Original','No comments','Rank'],...ab.map(r=>[r.model,r.macro_f1_original.toFixed(4),r.macro_f1_ablated.toFixed(4),`${r.rank_original} to ${r.rank_ablated}`])];
table(s,vals,60,214,745,290,[295,145,185,120],23,1);
text(s,'Data property',855,212,365,40,27,ink,true);
text(s,'Sparse comment and lexical fragments carry class cues. All four models lose F1.',855,267,365,132,27);
text(s,'SVM keeps first place. Comments alone do not explain its advantage.',855,425,365,121,27,ink,true);
text(s,`Prediction: SVM’s validation lead over Logistic Regression would shrink.\nObserved: ${d.svm_lr_validation_gap_before_pp.toFixed(2)} to ${d.svm_lr_validation_gap_after_pp.toFixed(2)} percentage points. Prediction not supported.`,60,552,1160,87,26);
text(s,'Exploratory follow-up. Forest/LR swap differs by only 0.00028 F1. Test ranks do not change.',60,661,1160,29,21,muted);
notes(s,'Sources: results/ablation_protocol.md, ablation_ranking.csv, ablation_interpretation.md and top_features.csv. The original SVM-only result was known before this extension, so the directional prediction is exploratory. All settings and split assignments stay fixed. One new training-only vectorizer supplies all four ablated classifiers. SVM test F1 falls from 0.579715 to 0.428673, a 15.1043 percentage point drop. The small forest/LR validation crossover is not evidence of a robust ordering. Test order stays SVM, Logistic Regression, Random Forest, Naive Bayes. Strong SVM features include comment openings by braces and inline comments by semicolons. Training density is 3.813%. Conclusion: comments carry useful information, but this ablation does not establish why SVM uniquely wins. Other lexical patterns may contribute; single-split evidence does not prove causality.');
const candidate=path.join(build,'candidate.pptx');await (await PresentationFile.exportPptx(pres)).save(candidate);
for(let i=0;i<pres.slides.items.length;i++){
 const slide=pres.slides.items[i];const png=await pres.export({slide,format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await slide.export({format:'layout'});await fs.writeFile(path.join(build,`slide-${i+1}.json`),await layout.text());
}
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(out,'LLM_AuthorBench_Assignment.pptx'),pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','4','--require-native-table-slide','5'],explicitTotalSlideCount:5,requiredNativeTableOwnerSlides:[2,4,5],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));

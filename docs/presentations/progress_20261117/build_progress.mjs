import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

// Run a copy in a private build folder whose node_modules points to the bundled runtime.
const workspace=path.dirname(fileURLToPath(import.meta.url));
const skill=process.env.PRESENTATIONS_SKILL_DIR || 'C:/Users/prakhar/.codex/plugins/cache/openai-primary-runtime/presentations/26.1116.12020/skills/presentations';
const python=process.env.RUNTIME_PYTHON || 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1116-1220-0_26.1116.1220.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
const {finalizePresentation,applyPresentationChartFont,resolvePresentationFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont({fontFamily:'Arial'});
const d=JSON.parse(await fs.readFile(path.join(workspace,'evidence.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#123047',teal:'#007F86',blue:'#477FA8',muted:'#536877',gray:'#A8B4BE',line:'#DCE4E9',light:'#F3F6F8',orange:'#C47538',white:'#FFFFFF'};
await fs.mkdir(path.join(workspace,'previews'),{recursive:true});
await fs.mkdir(path.join(workspace,'output'),{recursive:true});
await fs.mkdir(path.join(workspace,'.codex-finalizer'),{recursive:true});
const chartSlides=[],tableSlides=[];
function text(s,txt,x,y,w,h,size=24,color=C.navy,bold=false){
 const sh=s.shapes.add({name:txt.slice(0,65),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=[{runs:[{run:txt,style:{typeface:font,fontSize:`${size}px`,color,bold}}]}];
 sh.text.style={insets:0,wrap:'word',verticalAlignment:'top'};return sh;
}
function slide(title,sub='',source='Screenshot-reported Databricks results, 17 Nov 2026. Development validation, TEST disabled.'){
 const s=p.slides.add();s.background.fill=C.white;
 text(s,title,54,45,1172,100,42,C.navy,true);
 if(sub)text(s,sub,56,153,1165,54,23,C.muted);
 text(s,source,56,666,1110,32,14,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1180,666,48,25,15,C.muted);
 s.speakerNotes.text=source+' Clinical figures are transcribed from supplied screenshots and were not independently rerun locally. Multiple snapshots may refer to the same patient. All current validation estimates condition on prior model development and checkpoint selection.';
 return s;
}
function para(s,head,body,x,y,w=470){text(s,head,x,y,w,45,27,C.teal,true);text(s,body,x,y+51,w,110,24,C.navy);}
function table(s,rows,x,y,w,h,widths,fontsize=23){
 const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:rows});
 if([12,13].includes(p.slides.items.length)) t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({margins:{top:0,bottom:0,left:10,right:10}});
 t.borders.assign({fill:C.white,width:0});
 for(let r=0;r<rows.length;r++)for(let c=0;c<rows[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2===0?C.light:C.white);
  cell.text.style={fontFamily:font,fontSize:`${fontsize}px`,color:r===0?C.white:C.navy,bold:r===0,insets:10,verticalAlignment:'middle'};
 }
 tableSlides.push(p.slides.items.length);return t;
}
function chart(s,type,opts){const ch=s.charts.add(type,{chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},...opts});applyPresentationChartFont(ch,{fontFamily:font});chartSlides.push(p.slides.items.length);return ch;}
const ax={textStyle:{fontSize:20,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:{fill:C.line,width:1}};
function bar(s,cats,series,x,y,w,h,max=3.5,labels=true){return chart(s,'bar',{position:{left:x,top:y,width:w,height:h},categories:cats,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:series.length>1,legend:{position:'bottom',textStyle:{fontSize:19,fill:C.muted}},xAxis:{...ax,majorGridlines:null},yAxis:{...ax,min:0,max,numberFormatCode:max>50?'0':'0.0'},dataLabels:{showValue:labels,position:'outEnd',numberFormatCode:max>50?'0':'0.000',textStyle:{fontSize:21,bold:true,fill:C.navy}}});}

// 1
{
const s=p.slides.add();s.background.fill=C.navy;
text(s,'TAK861',62,70,1120,48,26,'#B7D9DE',true);
text(s,'Model development\nprogress review',60,204,1110,190,64,C.white,true);
text(s,'17 November 2026',64,438,1080,52,30,C.white);
text(s,'Current validation evidence and the next benchmark',64,510,1090,56,27,'#B7D9DE');
text(s,'Development results. Production comparison remains unresolved.',64,651,1100,30,18,'#D3E2E8');
s.speakerNotes.text='Progress review. Figures come from supplied notebook screenshots and local historical evidence. No claim of production superiority or untouched holdout performance.';
}
// 2
{
const s=slide('NLP fusion leads the current top-decile comparison','The evidence supports another bounded round of testing before a production decision');
text(s,'2.954×',60,228,430,90,70,C.teal,true);
text(s,'Lift at the top 10%\n+48 positive snapshots versus\nstatic LightGBM*',62,325,465,102,25,C.navy);
text(s,'*Derived from reported lift and capacity.',62,447,465,54,18,C.muted);
para(s,'+16.6% versus static LightGBM','The paired interval lower bound is just above zero. Replication on a later period remains necessary.',610,220,575);
para(s,'+44 versus fixed-budget static','279 versus 235 captures with 1,000-tree fits. Gain versus early-stopped LightGBM is untested.',610,407,575);
text(s,'Production readiness: exact-date scores and training provenance are still missing.',62,574,1138,58,25,C.navy,true);
}
// 3
{
const s=slide('What we tried','The work has progressed from a small cohort to reproducible comparisons on expanded histories');
table(s,[['Workstream','Experiment','What we learned'],['Representation','Counts, static fusion and temporal fusion on the original cohort','Temporal features were promising. Historical TEST remains retrospective.'],['Data expansion','48-month snapshot window, reconstructed initiation labels and patient splits','449,199 retained snapshots across 25,925 patients.'],['Current models','Count Transformer, monthly NLP fusion and static AutoML fusion','NLP fusion leads Lift@10% across the three supplied seeds.'],['Conventional model','Static LightGBM plus a matched timing-feature ablation','Adds capture to its fixed-budget control. Early-stopped protocol test pending.'],['Model combination','Historical fusion variants and a planned OOF blend','Blending did not help every configuration. Align scores before fitting a blend.']],56,222,1165,390,[215,440,510],21);
s.speakerNotes.text+=' Historical source: experiments/challenger/EVIDENCE.md and experiments/automl_sequence/HISTORICAL_ADDENDUM_FINDINGS.md. The expanded-cohort OOF blend has no supplied result.';
}
// 4
{
const s=slide('Expanded validation has 1,141 positive snapshots','Patient-disjoint splits share the same 48-month cutoff window. Snapshots exceed patient counts.');
text(s,'449,199',60,225,355,76,54,C.teal,true);text(s,'retained snapshots',62,303,360,42,25,C.muted);
text(s,'25,925',462,225,345,76,54,C.teal,true);text(s,'unique patients',465,303,350,42,25,C.muted);
text(s,'1.67%',872,225,340,76,54,C.teal,true);text(s,'positive snapshot rate',874,303,350,42,25,C.muted);
table(s,[['Split','Snapshots','Patients','Positive snapshots'],['TRAIN','315,343','18,193','5,166'],['Validation','67,969','3,919','1,141'],['Retrospective TEST','65,887','3,813','1,179']],56,373,1165,214,[350,260,230,325],23);
text(s,'Snapshot cutoffs: 19 Feb 2022–19 Jan 2026. TEST remains disabled for current development.',60,610,1150,43,21,C.muted);
s.speakerNotes.text+=' Cohort audit: 587,460 candidates, 449,209 inside the 48-month cutoff, 449,199 retained. Reconcile the latest label audit with the bound run before a final claim. Original eligibility producer and source completeness remain unverified. Clinical audit figures are screenshot-reported.';
}
// 5
{
const s=slide('Current inputs combine monthly histories with one AutoML anchor','SUMMARY uses one AutoML snapshot. A 12-anchor AutoML experiment is a separate next step.');
table(s,[['Input stream','Available tensor','Current representation'],['Counts','449,199 × 12 × 1,028','Monthly structured code counts'],['NLP','449,199 × 12 × 256','Monthly TF-IDF representation fitted on TRAIN'],['AutoML summary','449,199 × 1 × 511','One cutoff summary anchor in the current phase']],56,228,1165,265,[280,420,465],25);
text(s,'TRAIN-only selection',60,534,420,45,27,C.teal,true);
text(s,'Compact models use TRAIN-selected inputs. NLP uses a fixed TF-IDF vocabulary.',60,582,550,65,23,C.navy);
text(s,'Next representation question',670,534,545,45,27,C.teal,true);
text(s,'Test whether richer time structure and code-level pretraining add value under the same split.',670,582,545,65,23,C.navy);
s.speakerNotes.text+=' Source: supplied preparation receipts and current SUMMARY configuration. TF-IDF salience is not outcome importance. Step zero represents the most recent history window. Raw candidates, selected features and final tensor channels are distinct counts.';
}
// 6
{
const s=slide('NLP fusion reaches 2.954× lift at 10% capacity','Same 67,969 validation snapshots, 1,141 positives and 6,797 selected rows per model');
chart(s,'bar',{position:{left:56,top:230,width:825,height:360},categories:[...d.models].reverse().map(m=>m.name),series:[{name:'Validation Lift@10%',values:[...d.models].reverse().map(m=>m.lift10),fill:C.teal,points:[{idx:2,fill:C.blue},{idx:1,fill:'#7299B4'},{idx:0,fill:C.gray}]}],barOptions:{direction:'bar',gapWidth:65},hasLegend:false,xAxis:{...ax,majorGridlines:null},yAxis:{...ax,min:0,max:3.4,majorUnit:0.5,numberFormatCode:'0.0',title:'Lift at top 10%'},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:23,fill:C.navy,bold:true}}});
text(s,'+0.421',927,252,290,75,53,C.teal,true);text(s,'absolute lift versus\nstatic LightGBM',931,337,285,88,24,C.navy);
text(s,'+16.6%',927,445,290,75,49,C.teal,true);text(s,'relative lift improvement',931,526,285,64,23,C.navy);
text(s,'Static LightGBM is a representation control. Augmented models receive additional information.',60,616,1155,45,21,C.muted);
}
// 7
{
const s=slide('Only NLP’s 10% interval excludes zero','Paired patient-bootstrap intervals compare each model with the static LightGBM control');
table(s,[['Model','Lift difference','95% interval','Excludes zero'],['NLP fusion','+0.421','[+0.009, +0.802]','Yes'],['Static fusion','+0.254','[−0.059, +0.568]','No'],['Count Transformer','+0.202','[−0.204, +0.594]','No']],56,229,1165,252,[360,245,350,210],26);
para(s,'20% and 30% capacity','Every model-minus-LightGBM interval at these wider capacities includes zero.',60,525,545);
para(s,'Selection still matters','All four arms used validation for selection. These intervals do not establish an independent production gain.',670,525,545);
s.speakerNotes.text+=' Exact values: NLP delta .420677, CI [.009027,.801666]; static .254159, [-.059017,.568484]; count .201575, [-.203756,.593582]. At top20% and top30%, all supplied model-minus-LightGBM intervals include zero.';
}
// 8
{
const s=slide('Early stopping limits, but does not remove, overfitting','Training lift exceeds validation lift. Later epochs improve training while validation often stalls.');
const ms=[d.models[2],d.models[0],d.models[1]];
bar(s,ms.map(m=>m.name),[{name:'Training',values:ms.map(m=>m.train_lift10),fill:C.blue},{name:'Validation',values:ms.map(m=>m.lift10),fill:C.teal}],56,229,820,360,3.8);
text(s,'0.21–0.36',921,245,290,65,42,C.orange,true);text(s,'absolute train-to-validation\nlift gap at selected checkpoints',923,322,292,110,24,C.navy);
text(s,'7–11%',921,455,292,66,42,C.orange,true);text(s,'relative to training lift',923,523,292,53,23,C.muted);
text(s,'The results do not show catastrophic collapse. They also do not justify a claim of “no overfitting.”',60,616,1155,45,21,C.muted);
}
// 9
{
const s=slide('NLP fusion leads in all three reported seeds','Seed variation supports the ranking, but three seeds do not replace independent validation');
chart(s,'line',{position:{left:56,top:223,width:800,height:375},categories:['Seed 42','Seed 142','Seed 242'],series:[{name:'NLP fusion',values:d.seed_lifts.NLP_FUSION,line:{fill:C.teal,width:4},marker:{symbol:'circle',size:9,fill:C.teal}},{name:'Static fusion',values:d.seed_lifts.STATIC_FUSION,line:{fill:C.blue,width:3},marker:{symbol:'circle',size:8,fill:C.blue}},{name:'Count Transformer',values:d.seed_lifts.COUNT,line:{fill:C.gray,width:3},marker:{symbol:'circle',size:8,fill:C.gray}}],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:20}},xAxis:{...ax,majorGridlines:null},yAxis:{...ax,min:2.5,max:3.15,numberFormatCode:'0.00',title:'Validation Lift@10%'},dataLabels:{showValue:false}});
text(s,'Selected epochs',900,242,305,45,27,C.teal,true);
text(s,'NLP fusion: 2–4\nStatic fusion: 3–4\nCount: 3–12',902,305,307,156,26,C.navy);
text(s,'The final comparison uses averaged predictions. It is not the mean of these seed lifts.',902,492,307,111,23,C.muted);
s.speakerNotes.text+=' STATIC_FUSION seed 42 and the ensemble both report Lift@10%=2.786988 in the supplied notebook 03 and 04 outputs. Rechecked against first attachment batch Photo 1 and Photo 5. The discrete capture metric can be equal for different score vectors, so the reported value is retained.';
}
// 10
{
const s=slide('Timing features add 44 positive captures','Paired LightGBM ablation with identical validation rows and a fixed TRAIN-only tree budget');
bar(s,['Static','Static + time'],[{name:'Positive snapshots captured',values:[235,279],fill:C.teal,points:[{idx:0,fill:C.gray}]}],56,235,620,356,320);
text(s,'+44',733,236,450,88,65,C.teal,true);text(s,'positive snapshots at 10% capacity',737,329,458,67,27,C.navy);
text(s,'2.060×  to  2.445×',737,424,458,58,36,C.navy,true);
text(s,'Lift difference +0.386\n95% interval [+0.098, +0.689]',737,500,460,83,25,C.navy);
text(s,'Timing adds signal to the fixed-budget fit. Gain under the early-stopped NB05 protocol remains untested.',60,614,1150,45,22,C.muted);
s.speakerNotes.text+=' Features use monthly sleep-study procedure and sleep-wake diagnosis histories. STATIC_PLUS_TIME captured279/464/624 at10/20/30%, versus235/427/571. At20% delta.162136CI[-.004921,.325341]; at30% delta.154833CI[.032278,.262836]. 1,000-tree fixed TRAIN-only budget. Counts are snapshots, not unique patients.';
}
// 11
{
const s=slide('Production benchmarking is blocked by score coverage','The current production table has no exact patient-date match for the expanded validation rows');
text(s,'0 / 67,969',60,246,550,100,64,C.orange,true);
text(s,'validation snapshots with an\nexact-key production score',62,356,544,90,29,C.navy);
text(s,'No production lift is available\nfor this comparison.',62,510,545,80,28,C.navy,true);
para(s,'Required scoring evidence','Frozen model and input definitions, historical feature vintage, exact cutoff dates and complete score coverage.',690,229,525);
para(s,'Required training evidence','Training cutoff and patient membership, plus target alignment. Same-row scoring alone cannot establish independence.',690,431,525);
s.speakerNotes.text+=' The supplied notebook05 prints all67969missing exact keys, PRODUCTION_FROZEN_SCORES=NaN and BLOCKED. Existing models still have controlled development results; NOT_COMPARABLE pertains to production comparison. No nearest-date or intersection fallback. Historical4.592score was a different population and has unverified fit overlap.';
}
// 12
{
const s=slide('Next steps prioritize signal and a valid decision','Advance the strongest evidence while preserving a later-period confirmation set');
table(s,[['Priority','Next experiment or dependency','Decision criterion'],['1','Align production scores and training provenance','Same population and known fit membership'],['2','Retest timing in NB05, then fusion / 12-anchor AutoML','Gain on identical keys with paired uncertainty'],['3','Bounded Med-BERT-style pretraining benchmark','Pretraining gain versus scratch and compute used'],['4','Fit one blend using TRAIN out-of-fold scores','Beat both parents on the same validation rows'],['5','Freeze one candidate for a later-period test','Mature outcomes and a pre-agreed capture gain']],56,229,1165,367,[130,535,500],23);
text(s,'Source completeness, eligibility provenance and treatment/claim-status definitions remain evidence checks.',60,616,1155,45,21,C.muted);
s.speakerNotes.text+=' Confirm label-correction timing against the bound run. Secondary commentary is not a new target-change instruction. Zolpidem and reversed/rejected-claim concerns from secondary commentary need source/target verification before any label change. If timing adds little, prior therapy changes/diversity is a separate predeclared fallback. Blend weights must come from patient-grouped out-of-fold TRAIN predictions. Transcript commitments: settings11:05, sequentialAutoML12:24–12:45, Med-BERT13:07–13:25, later-period confirmation13:46–14:04.';
}
// 13
{
const s=slide('Med-BERT-style benchmark will test code pretraining','Official pretrained weights are unavailable. This experiment adapts the method to monthly histories.');
table(s,[['Stage','Implementation intent','Evidence returned'],['Input contract','Code-presence tokens from monthly counts','Up to 2,000 TRAIN patients / 25,000 snapshots'],['Training comparison','TRAIN masked pretraining, then fine-tuning','One seed. Matched scratch control.'],['Validation','Score the same complete validation keys','Lift@10/20/30, captures, AP and paired uncertainty'],['Runtime control','Cached tensors with a 55-minute budget','Elapsed time and completion status']],56,229,1165,315,[235,490,440],23);
text(s,'One-hour target depends on load time and hardware. Monthly tokens differ from original Med-BERT inputs. Existing benchmark references used full TRAIN.',60,580,1150,71,24,C.navy);
s.speakerNotes.text+=' Planned benchmark, no measured Med-BERT lift. A compact monthly code-token adaptation differs from the published structured-EHR Med-BERT and must be labelled accordingly. Current TF-IDF or aggregated count vectors cannot simply be passed into a pretrained diagnosis vocabulary. Checkpoint use needs a verified compatible vocabulary and architecture. Authoritative background: Rasmy et al., Med-BERT, npj Digital Medicine (2021), https://www.nature.com/articles/s41746-021-00455-y and https://github.com/ZhiGroup/Med-BERT.';
}
// 14
{
const s=slide('Historical results motivate temporal features, with limits','Original validation is a separate cohort from today’s expanded results','Historical screenshot-reported results and local EVIDENCE.md. Comparisons belong to their original split.');
bar(s,d.historical.old_validation.models,[{name:'Validation Lift@10%',values:d.historical.old_validation.lift10,fill:C.blue,points:[{idx:2,fill:C.teal}]}],56,230,820,359,3.2);
text(s,'3,481',931,243,290,70,49,C.teal,true);text(s,'validation snapshots\n202 positive snapshots',933,323,290,90,25,C.navy);
text(s,'2.716 vs 2.568',900,459,325,55,32,C.navy,true);text(s,'Temporal fusion versus\nstatic LightGBM',903,525,320,65,23,C.muted);
text(s,'The rounded results imply three extra positive captures at 349-slot capacity. No paired interval was supplied.',60,615,1155,46,21,C.muted);
}
// 15
{
const s=slide('Historical TEST did not resolve the model ranking','Retrospective TEST had prior inspection and upstream feature-selection limitations','Historical screenshot-reported results and local EVIDENCE.md. This is not an untouched holdout.');
table(s,[['Model on original retrospective TEST','Lift@10%','Interpretation'],['Experimental AutoML LightGBM','3.163','64 implied captures'],['Current temporal fusion','3.113','63 implied captures'],['Warm-start static fusion','3.113','Distinct model, same rounded lift'],['Logistic late fusion','3.064','Alternative combination'],['Current static fusion / count','2.718 / 2.372','Same current model family']],56,228,1165,310,[550,190,425],23);
text(s,'Temporal minus LightGBM: −0.049, paired 95% interval [−0.645, +0.599]',60,578,1150,48,27,C.navy,true);
text(s,'The interval leaves the ranking unresolved. It does not establish equivalence.',60,624,1155,34,21,C.muted);
s.speakerNotes.text+=' Original retrospectiveTEST3414snapshots202positives. Earlier representation+AutoMLLightGBMhead2.669versusrepresentationalone2.817showscombinationdoesnotalwayshelp. Earlier lockedcount2.866isadifferentrunfromcurrentcount2.372. Historicalproduction4.592andV63-styleTRAINrefit2.52belongtoanothercontrastandhaveunverifiedtrainingoverlap. No direct link from these scores to the expanded benchmark.';
}
// 16
{
const s=slide('Current metrics and capacity trade-offs','Static fusion leads at 20% and 30%. NLP ties static LightGBM at 30% capacity.');
table(s,[['Model','Lift@10%','Lift@20%','Lift@30%','AP','ROC-AUC'],...d.models.map(m=>[m.name,m.lift10.toFixed(3),m.lift20.toFixed(3),m.lift30.toFixed(3),m.ap?m.ap.toFixed(4):'—',m.auc?m.auc.toFixed(4):'—'])],56,230,1165,280,[360,165,165,165,155,155],24);
text(s,'Metric definition',60,547,510,40,27,C.teal,true);
text(s,'Lift equals the positive rate in the selected top fraction divided by the overall positive rate.',60,600,550,60,22,C.navy);
text(s,'Interpretation',680,547,510,40,27,C.teal,true);
text(s,'Derived captures: NLP 509 versus static fusion 512 at 20%, then both NLP and LightGBM 650 at 30%.',680,594,525,50,21,C.navy);
text(s,'Captures derive from reported lift and capacity.',680,647,525,20,15,C.muted);
s.speakerNotes.text+=' *Captures derive from reported lift and capacity, rounded to the consistent integer counts. AP and AUC for static LightGBM are unavailable in the supplied comparison and remain blank. Validation prevalence is 1141/67969=1.6787%. Exact scores and patient-cluster bootstrap are needed for new paired comparisons. Low AP needs prevalence context. Captures are snapshots. Unique-patient contact decisions need patient-level reporting.';
}

for(let i=0;i<p.slides.items.length;i++){
const s=p.slides.items[i]; const png=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(path.join(workspace,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
await fs.writeFile(path.join(workspace,'previews',`slide-${String(i+1).padStart(2,'0')}.layout.json`),await (await s.export({format:'layout'})).text());
}
const candidate=path.join(workspace,'.codex-finalizer','candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(workspace,'output',process.env.DECK_FILENAME||'TAK861_Progress_20261117.pptx');
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath:final,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableSlides)].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[...new Set(tableSlides)],requiredNativeChartOwnerSlides:[...new Set(chartSlides)],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(workspace,'.codex-finalizer',path.basename(final)+'.validation.json')});
console.log(JSON.stringify({final,slides:p.slides.items.length,chartSlides,tableSlides}));

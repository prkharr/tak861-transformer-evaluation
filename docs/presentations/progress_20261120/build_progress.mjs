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
const specs=JSON.parse(await fs.readFile(path.join(workspace,'slides.json'),'utf8'));
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
function slide(title,sub='',source='Development results | 20 November 2026'){
 const s=p.slides.add();s.background.fill=C.white;
 text(s,title,54,45,1172,100,42,C.navy,true);
 if(sub)text(s,sub,56,153,1165,54,23,C.muted);
 text(s,source,56,666,1110,32,14,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1180,666,48,25,15,C.muted);
 s.speakerNotes.text='Development validation; multiple snapshots may refer to the same patient.';
 return s;
}
function para(s,head,body,x,y,w=470){text(s,head,x,y,w,45,27,C.teal,true);text(s,body,x,y+51,w,110,24,C.navy);}
function table(s,rows,x,y,w,h,widths,fontsize=23){
 const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:rows});
 if(true) t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({margins:{top:0,bottom:0,left:10,right:10}});
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


for (const q of specs) {
 const s=q.kind==='cover'?p.slides.add():slide(q.title,q.sub,q.note||'Development results | 20 November 2026');
 s.speakerNotes.text=q.note||q.sub;
 if(q.kind==='cover') {
   s.background.fill=C.navy;

   text(s,'TAK861',64,70,1100,50,28,'#B7D9DE',true);
   text(s,'Deep learning\nproof of concept',60,215,1140,180,64,C.white,true);
   text(s,'Progress and development priorities',64,448,1100,65,32,C.white);
   text(s,'20 November 2026',64,565,1100,50,26,'#B7D9DE');
 } else if(q.kind==='columns') {
   q.items.forEach((item,i)=>{
     const x=60+i*400;
     text(s,item[0],x,240,365,80,28,C.teal,true);
     text(s,item[1],x,340,365,245,26,C.navy);
   });
 } else if(q.kind==='table') {
   table(s,q.rows,56,225,1165,q.takeaway?310:390,q.widths,22);
 } else if(q.kind==='chart') {
   bar(s,q.categories,[{name:q.metric,values:q.values,fill:C.teal}],80,230,1110,320,q.max);
 }
 if(q.takeaway)text(s,q.takeaway,60,570,1155,80,26,C.navy,true);
}
for(let i=0;i<p.slides.items.length;i++){
const s=p.slides.items[i]; const png=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(path.join(workspace,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
await fs.writeFile(path.join(workspace,'previews',`slide-${String(i+1).padStart(2,'0')}.layout.json`),await (await s.export({format:'layout'})).text());
}
const candidate=path.join(workspace,'.codex-finalizer','candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(workspace,'output',process.env.DECK_FILENAME||'TAK861_Progress_20261120.pptx');
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath:final,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableSlides)].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[...new Set(tableSlides)],requiredNativeChartOwnerSlides:[...new Set(chartSlides)],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(workspace,'.codex-finalizer',path.basename(final)+'.validation.json')});
console.log(JSON.stringify({final,slides:p.slides.items.length,chartSlides,tableSlides}));


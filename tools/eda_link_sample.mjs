import fs from 'node:fs';
import readline from 'node:readline';
import path from 'node:path';

const root = process.argv[2] ?? 'dataset';
const limit = Number(process.argv[3] ?? 5000);
const normalize = s => (s ?? '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim().replace(/\s+/g, ' ');
const tokens = s => new Set(normalize(s).split(' ').filter(t => t.length >= 2));
const overlap = (a,b) => { let n=0; for (const t of a) if (b.has(t)) n++; return n / Math.max(1, a.size+b.size-n); };
async function each(file, cb) {
  const rl=readline.createInterface({input:fs.createReadStream(file,{encoding:'utf8'}),crlfDelay:Infinity});
  let first=true; for await(const line of rl) { if(first){first=false;continue;} const c=line.split('\t'); await cb(c); }
}
const gtFile=path.join(root,'train','train_ground_truth.tsv');
const s1Ids=new Set(), targetIds=new Set();
const labelN={rows:0, links:0, singletons:0, targets:{s2:0,s3:0}};
await each(gtFile, c=>{
  if(labelN.rows>=limit)return;
  labelN.rows++; s1Ids.add(c[0]);
  const ids=(c[1]??'').split(',').filter(Boolean); labelN.links+=ids.length;
  if(!ids.length) labelN.singletons++;
  for(const id of ids){targetIds.add(id);if(id.startsWith('S2-'))labelN.targets.s2++;else if(id.startsWith('S3-'))labelN.targets.s3++;}
});
const records=new Map();
for(const [source,file] of [['s1','train_source1.tsv'],['s2','train_source2.tsv'],['s3','train_source3.tsv']]){
  await each(path.join(root,'train',file),c=>{
    const id=c[0]; if((source==='s1'?s1Ids:targetIds).has(id)) records.set(id,{source,id,name:c[1]??'',address:c[2]??'',country:c[3]??''});
  });
  console.log(`scanned ${source}; found sampled records ${records.size.toLocaleString()}`);
}
const totals={pairs:0,missingS1:0,missingTarget:0,nameExact:0,addressExact:0,countryDiff:0,nameToken:0,addressToken:0,nameBothTokens:0,addressDigitOverlap:0};
const nameJ=[],addrJ=[];
await each(gtFile,c=>{
  if(!s1Ids.has(c[0]))return;
  const a=records.get(c[0]); if(!a){totals.missingS1++;return;}
  for(const id of (c[1]??'').split(',').filter(Boolean)){
    const b=records.get(id); if(!b){totals.missingTarget++;continue;}
    totals.pairs++;
    const an=normalize(a.name),bn=normalize(b.name),aa=normalize(a.address),ba=normalize(b.address);
    if(an===bn)totals.nameExact++;
    if(aa===ba)totals.addressExact++;
    if(a.country!==b.country)totals.countryDiff++;
    const nt=overlap(tokens(a.name),tokens(b.name)),at=overlap(tokens(a.address),tokens(b.address));
    nameJ.push(nt);addrJ.push(at); if(nt>0)totals.nameToken++;if(at>0)totals.addressToken++;
    if(nt>0&&at>0)totals.nameBothTokens++;
    const digs=x=>new Set((x.match(/\d{4,}/g)??[]));
    const ad=digs(a.address),bd=digs(b.address); if([...ad].some(x=>bd.has(x)))totals.addressDigitOverlap++;
  }
});
const quant=x=>{x.sort((a,b)=>a-b);const at=p=>x[Math.min(x.length-1,Math.floor(p*x.length))]??null;return {p10:at(.1),median:at(.5),p90:at(.9),p99:at(.99)}};
console.log(JSON.stringify({sampledEntities:labelN,foundRecordCount:records.size,pairCoverage:totals.pairs,totals,positivePairSimilarity:{nameTokenJaccard:quant(nameJ),addressTokenJaccard:quant(addrJ)}},null,2));

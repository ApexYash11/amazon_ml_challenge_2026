import fs from 'node:fs';
import readline from 'node:readline';
import path from 'node:path';

const root = process.argv[2] ?? 'dataset';
const files = [
  ['train_s1', path.join(root, 'train', 'train_source1.tsv')],
  ['train_s2', path.join(root, 'train', 'train_source2.tsv')],
  ['train_s3', path.join(root, 'train', 'train_source3.tsv')],
  ['train_gt', path.join(root, 'train', 'train_ground_truth.tsv')],
  ['test_s1', path.join(root, 'test', 'test_source1.tsv')],
  ['test_s2', path.join(root, 'test', 'test_source2.tsv')],
  ['test_s3', path.join(root, 'test', 'test_source3.tsv')],
];

function blankStats() {
  return { rows: 0, header: [], empty: {}, country: new Map(), lengths: { name: [], address: [] },
    exactName: new Set(), exactAddress: new Set(), duplicateRows: { name: 0, address: 0 }, sampleRows: 0 };
}
function normalized(s) { return (s ?? '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim().replace(/\s+/g, ' '); }
function quantiles(xs) {
  if (!xs.length) return {};
  xs.sort((a,b)=>a-b);
  const at = p => xs[Math.min(xs.length - 1, Math.floor(p * xs.length))];
  return { min: xs[0], p25: at(.25), median: at(.5), p75: at(.75), p90: at(.9), max: xs.at(-1) };
}
async function eachLine(file, cb) {
  const rl = readline.createInterface({ input: fs.createReadStream(file, { encoding: 'utf8' }), crlfDelay: Infinity });
  let first = true;
  for await (const line of rl) {
    if (first) { first = false; continue; }
    if (line) await cb(line.split('\t'));
  }
}
function bump(map, key) { map.set(key, (map.get(key) ?? 0) + 1); }

const stats = new Map();
for (const [name, file] of files) {
  if (!fs.existsSync(file)) throw new Error(`Missing file: ${file}`);
  const st = blankStats();
  const first = fs.readFileSync(file, { encoding: 'utf8', flag: 'r' }).slice(0, 4096).split(/\r?\n/, 1)[0];
  st.header = first.split('\t');
  const counts = Array(st.header.length).fill(0);
  await eachLine(file, cols => {
    st.rows++;
    for (let i=0; i<st.header.length; i++) if (!(cols[i] ?? '').trim()) counts[i]++;
    if (st.header.includes('country')) bump(st.country, (cols[st.header.indexOf('country')] ?? '').trim() || '<empty>');
    if (st.header.includes('business_name')) {
      const n = normalized(cols[st.header.indexOf('business_name')]);
      const a = normalized(cols[st.header.indexOf('business_address')]);
      if (st.sampleRows < 100000) {
        if (n) { st.lengths.name.push(n.length); if (st.exactName.has(n)) st.duplicateRows.name++; else st.exactName.add(n); }
        if (a) { st.lengths.address.push(a.length); if (st.exactAddress.has(a)) st.duplicateRows.address++; else st.exactAddress.add(a); }
        st.sampleRows++;
      }
    }
  });
  st.empty = Object.fromEntries(st.header.map((h,i)=>[h, counts[i]]));
  stats.set(name, st);
  console.log(`${name}: ${st.rows.toLocaleString()} rows; columns=${st.header.join(',')}; empty=${JSON.stringify(st.empty)}; countries=${JSON.stringify(Object.fromEntries([...st.country].sort((a,b)=>b[1]-a[1])))}; normalized_text_lengths=${JSON.stringify({name:quantiles(st.lengths.name),address:quantiles(st.lengths.address)})}`);
  if (name.startsWith('train_') && name !== 'train_gt') {
    console.log(`  first-100k-row normalized duplicate count: name=${st.duplicateRows.name.toLocaleString()}, address=${st.duplicateRows.address.toLocaleString()}`);
  }
}

const gt = stats.get('train_gt');
const gtPath = path.join(root, 'train', 'train_ground_truth.tsv');
const hist = new Map();
const sourceCount = { s2: 0, s3: 0, invalid: 0 };
let checkedS1Labels = 0;
await eachLine(gtPath, cols => {
  const s1 = cols[0] ?? '';
  const ids = (cols[1] ?? '').split(',').map(x=>x.trim()).filter(Boolean);
  bump(hist, ids.length);
  if (s1) checkedS1Labels++;
  for (const id of ids) {
    if (id.startsWith('S2-')) sourceCount.s2++;
    else if (id.startsWith('S3-')) sourceCount.s3++;
    else sourceCount.invalid++;
  }
});
console.log(`ground_truth: rows=${checkedS1Labels.toLocaleString()}; cardinality_histogram=${JSON.stringify(Object.fromEntries([...hist].sort((a,b)=>a[0]-b[0])))}; source_link_counts=${JSON.stringify(sourceCount)}; invalid_prefix_links=${sourceCount.invalid}`);
console.log(`train-test country overlap: ${JSON.stringify({ train: [...new Set([...stats.get('train_s1').country.keys(), ...stats.get('train_s2').country.keys(), ...stats.get('train_s3').country.keys()])], test: [...new Set([...stats.get('test_s1').country.keys(), ...stats.get('test_s2').country.keys(), ...stats.get('test_s3').country.keys()])] })}`);

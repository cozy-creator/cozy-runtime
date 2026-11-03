from pathlib import Path
import base64,csv,hashlib,io,json,subprocess,zipfile
root=Path.cwd();receipt=json.loads(Path('/home/fidika/cozy_v2/outputs/varena-v2-p0-20261031/runtime-candidate-88b7f761.json').read_text());native=json.loads(Path(receipt['native_reuse_receipt']).read_text());wheel=Path(receipt['wheel']);out=root/'.task/runtime-wheel-003';assert not out.exists()
def sha(data):return hashlib.sha256(data).hexdigest()
assert sha(wheel.read_bytes())==receipt['sha256']
with zipfile.ZipFile(wheel) as z:
 names=z.namelist();assert len(set(names))==len(names)
 record,=[n for n in names if n.endswith('.dist-info/RECORD')];rows=list(csv.reader(io.StringIO(z.read(record).decode())));assert {r[0] for r in rows}=={n for n in names if not n.endswith('/')}
 for name,digest,size in rows:
  if name==record:assert digest==size=='';continue
  data=z.read(name);assert digest=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=');assert len(data)==int(size)
 native_hashes={name:sha(z.read(name)) for name in native['files']};assert native_hashes==native['files'];assert len(native_hashes)==581
 source_hashes={}
 for name in subprocess.check_output(['git','ls-tree','-r','--name-only',receipt['source'],'--','src/cozy_runtime'],text=True).splitlines():
  p=Path(name)
  if p.suffix not in ('.py','.pyi') or p.name=='_build_provenance.py':continue
  member=str(p.relative_to('src'));data=subprocess.check_output(['git','show',receipt['source']+':'+name]);assert data==z.read(member),name;source_hashes[member]=sha(data)
 assert receipt['source'] in z.read('cozy_runtime/_build_provenance.py').decode()
 z.extractall(out)
x={'passed':True,'wheel':str(wheel),'wheel_sha256':receipt['sha256'],'source':receipt['source'],'version':receipt['version'],'record_rows':len(rows),'record_complete':True,'native_members':len(native_hashes),'all_native_sha256_equal_a09':True,'source_members':len(source_hashes),'source_members_sha256':source_hashes,'isolated_extraction':str(out),'native_rebuilt':False,'gpu_qualified':False}
(root/'.task/runtime-wheel-closure-003.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='source_members_sha256'}))

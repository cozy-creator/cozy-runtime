from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path.cwd();previous=json.loads((root/'docs/evidence/current-cohort-118d/runtime-native-reuse-002.json').read_text());current=json.loads((root/'.task/native-spec-current-003.json').read_text());frozen=json.loads((root/'docs/evidence/current-cohort-118d/native-spec-frozen-002.json').read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
assert current['sets']==frozen['sets']
assert current['launcher_source_sha256']==frozen['launcher_source_sha256']
inputs=[]
for row in previous['source_input_files']:
 digest=sha((root/row['path']).read_bytes());assert digest==row['current_sha256'],row['path'];inputs.append({'path':row['path'],'sha256':digest})
native=json.loads(Path('/home/fidika/cozy_v2/outputs/xs138-p2-vae-block/native-a09e68dba/native-build-receipt.json').read_text());controlled=json.loads(Path('/home/fidika/cozy_v2/outputs/xs138-vae-immutable/controlled-wheel-001/receipt.json').read_text());expected=native['files'];assert len(expected)==581
wheels={}
for label,path in [('base',Path(controlled['base_wheel'])),('controlled',Path(controlled['wheel']))]:
 digest=sha(path.read_bytes());assert digest==previous['wheels'][label]['sha256']
 with zipfile.ZipFile(path) as z:
  assert {name:sha(z.read(name)) for name in expected}==expected
  manifest=json.loads(z.read('cozy_runtime/_triton/manifest.json'))
 wheels[label]={'path':str(path),'sha256':digest,'native_members_verified':len(expected)}
assert {p:sha((Path(native['prefix'])/p).read_bytes()) for p in expected}==expected
keys={key for row in current['sets'] for key in row['entries'].values()};assert keys==set(manifest['kernels']);assert {r['driver_sha256'] for r in manifest['launcher'].values()}=={current['launcher_source_sha256']}
assert not subprocess.check_output(['git','diff','--name-only','83ce9027e','HEAD','--','pyproject.toml','uv.lock'],text=True).strip()
r={'passed':True,'runtime_source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'previous_receipt':'docs/evidence/current-cohort-118d/runtime-native-reuse-002.json','source_input_count':len(inputs),'source_inputs':inputs,'changed_native_inputs':[],'triton_version':current['triton'],'spec_pairs':len(current['sets']),'triton_keys':len(keys),'all_keys_match_frozen_and_manifest':True,'launcher_sha256':current['launcher_source_sha256'],'native_members':len(expected),'prefix_verified':native['prefix'],'wheels':wheels,'pyproject_and_uv_lock_unchanged_from_previous_freeze':True,'cuda_initialized':False,'decision':'reuse existing581 native members; build separately stamped current Runtime wheel from88b7 source; no native rebuild'}
(root/'.task/runtime-native-reuse-003.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_inputs','wheels']}))

from pathlib import Path
import json,os,subprocess,time,hashlib
root=Path.cwd();site=root/'.task/runtime-wheel-003';cpu=Path('/home/fidika/cozy/.worktrees/cozy-creator/xs138-p4-torch-offload/.task/torch-cpu');tfsbase=Path('/home/fidika/cozy_v2/outputs/xs138-native-api-integrated/linux-d3e8b3651-final-001');tfssite=tfsbase/'wheel-tree';candidate=json.loads((tfsbase/'candidate.json').read_text());assert hashlib.sha256(Path(candidate['wheel_path']).read_bytes()).hexdigest()==candidate['wheel_sha256'];assert hashlib.sha256((tfssite/'tensorfs/_ext.abi3.so').read_bytes()).hexdigest()==candidate['extension_sha256'];assert hashlib.sha256((tfssite/'tensorfs/.native/libvarena.so').read_bytes()).hexdigest()==candidate['core_sha256']
env=dict(os.environ,PYTHONPATH=':'.join([str(site),str(tfssite),'/home/fidika/cozy_v2/cozy-runtime/.venv/lib/python3.12/site-packages']),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',COZY_HOME=str(root/'.task/runtime-wheel-cpu-home-003'))
probe='''import json,sys
from pathlib import Path
import torch,cozy_runtime
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.internal.execution import cuts,native_refiner_ffn
from tensorfs import _ext
assert cozy_runtime.__version__ == '0.37.2+dev.g88b7f761'
assert COMMIT == '88b7f7619a8d1669bd8ae89e9e285a302ac3c061'
assert '/.task/runtime-wheel-003/' in cozy_runtime.__file__ and '/.task/runtime-wheel-003/' in cuts.__file__ and '/.task/runtime-wheel-003/' in native_refiner_ffn.__file__
assert not torch.cuda.is_initialized() and 'libcuda.so' not in Path('/proc/self/maps').read_text()
print(json.dumps({'passed':True,'runtime':cozy_runtime.__file__,'version':cozy_runtime.__version__,'commit':COMMIT,'refiner':native_refiner_ffn.__file__,'tensorfs_extension':_ext.__file__,'torch':torch.__version__,'cuda_initialized':False,'driver_loaded':False}))'''
strict=(root/'research/varena-multi-gpu-start/cpu.py').read_text();needle="runtime = Path(__file__).resolve().parents[2] / 'src'";assert strict.count(needle)==1;strict=strict.replace(needle,'runtime = Path('+repr(str(site))+')');(root/'.task/strict-wheel-startup-003.py').write_text(strict)
frames=Path('/home/fidika/cozy/.worktrees/cozy-creator/xs138-v2-current-core/crates/varena/evidence/current-cohort-118d/sender-frames-002.json')
commands=[('wheel-import-003',[str(cpu/'bin/python'),'-c',probe]),('wheel-cli-003',[str(cpu/'bin/python'),'-c','from cozy_runtime.cli.main import main; raise SystemExit(main(["--json","version"]))']),('refiner-host-003',[str(cpu/'bin/python'),'research/varena-refiner-ffn/host_check.py']),('refiner-semantic-003',[str(cpu/'bin/python'),'research/varena-refiner-ffn/semantic_check.py']),('strict-wheel-startup-003',['/home/fidika/cozy_v2/cozy-runtime/.venv/bin/python','-I',str(root/'.task/strict-wheel-startup-003.py'),'--tensorfs-python',str(tfssite),'--torch-site',str(cpu/'lib/python3.12/site-packages'),'--sender-frames',str(frames),'--out',str(root/'.task/strict-wheel-startup-003')]),('refiner-fixture-import-003',[str(cpu/'bin/python'),'-c',"import sys;sys.path.insert(0,'research/varena-refiner-ffn/fixture');import ci_varena_refiner_ffn;print('updated actual-dispatch fixture imported; no GPU execution')"])]
results=[]
for name,cmd in commands:
 command=['nice','-n','19',*cmd];start=time.time()
 with (root/'.task'/f'{name}.log').open('x') as out:p=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT)
 row={'name':name,'command':command,'exit':p.returncode,'seconds':round(time.time()-start,3)};results.append(row);(root/'.task/wheel-cpu-checks-003.json').write_text(json.dumps({'results':results,'runtime_site':str(site),'tensorfs_candidate':candidate['wheel_path'],'tensorfs_sha256':candidate['wheel_sha256'],'source_runtime':str(root),'gpu_qualified':False},indent=2)+'\n');print(json.dumps(row),flush=True)
 if p.returncode:raise SystemExit(p.returncode)

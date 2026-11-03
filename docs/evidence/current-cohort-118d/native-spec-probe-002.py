from pathlib import Path
import hashlib,json,sys
source=Path(sys.argv[1]);out=Path(sys.argv[2])
sys.path[:0]=[str(source/'src'),'/home/fidika/cozy_v2/outputs/xs138-native-tensor-client/path-proof-001/site','/home/fidika/cozy/.worktrees/cozy-creator/xs138-p0-torch/.task/torch-sdk/2.14.0+cu130-08187d9e/site-packages']
import torch,triton
from cozy_runtime.internal import triton_prebuild,triton_kernels
assert not torch.cuda.is_initialized()
rows=[]
for kernels,arch in triton_prebuild.sets():
 spec=triton_kernels.spec(kernels,arch);spec.pop('shipped');rows.append({'name':kernels.name,**spec})
r={'source':str(source),'torch':torch.__version__,'triton':triton.__version__,'sets':rows,'launcher_source_sha256':hashlib.sha256(triton_kernels.launcher_source()).hexdigest(),'cuda_initialized':torch.cuda.is_initialized()}
assert not r['cuda_initialized'];out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'source':str(source),'sets':len(rows),'triton':triton.__version__,'unique_keys':len({k for r in rows for k in r['entries'].values()}),'cuda_initialized':False}))

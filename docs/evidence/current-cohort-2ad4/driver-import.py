import json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2],'/home/fidika/cozy/.worktrees/cozy-creator/xs138-p0-torch/.task/torch-sdk/2.14.0+cu130-08187d9e/site-packages']
def snapshot():
 return sorted({line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines() if ('libcuda.so' in line or 'libnvidia-ml.so' in line)})
result={'before':snapshot()}
import tensorfs
result['tensorfs']=snapshot()
import torch
result['torch']=snapshot()
from cozy_runtime.internal import executor,cooperation
result.update(executor=snapshot(),cuda_initialized=torch.cuda.is_initialized(),capable=cooperation.capable(),tensorfs_file=tensorfs.__file__,torch_file=torch.__file__)
print(json.dumps(result,indent=2))

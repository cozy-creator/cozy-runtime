"""CPU semantic-descriptor regression using actual failed GPU algorithm metadata."""
from pathlib import Path
import ast,copy,hashlib,json,sys
from dataclasses import asdict,replace
from types import SimpleNamespace as N
out=Path(__file__).resolve().parent
repo=Path('/home/fidika/cozy/.worktrees/cozy-runtime/xs138-p2-vae-immutable')
sys.path[:0]=[str(repo/'src'),'/home/fidika/cozy_v2/outputs/xs138-p4-public-consumer/helper-2990db724-parity004-001/host-overlay','/home/fidika/cozy/.worktrees/cozy-creator/xs138-p0-torch/.task/torch-sdk/2.14.0+cu130-08187d9e/site-packages','/home/fidika/cozy_v2/cozy-runtime/.venv/lib/python3.12/site-packages']
import torch
from cozy_runtime.internal import dense_linear
from cozy_runtime.internal.execution import phase
assert not torch.cuda.is_initialized()
receipt=Path('/home/fidika/cozy_v2/outputs/varena-v2-p0-20261031/native-api-source-leg17-plan/attempt-005/results/vae-full-result.json')
record=json.loads(receipt.read_text())
queries=record['cases'][0]['queries']
algorithms=tuple(((q['rows'],q['inputs'],q['outputs']),q['algorithm'],q['workspace_bytes'],q['library_version'],q['header_version']) for q in queries)
q=queries[0]
policy=dense_linear.Policy(q['workspace_limit_bytes'],q['half_accumulation'],q['reduction'],q['sm_target'],'Cublas',None)
# Remaining fields are immutable metadata of the admitted producer layout. No fake
# CUDA calls, tensor operations or memory-demand assertions are made by this check.
layout=((3,1797,2048),(3680256,2048,1),'torch.float16',0)
inventory=N(site=N(layout=N(kernels=N(identity='actual-vae-block-kernel-contract')),eps=1e-6,qk=(N(eps=1e-6),N(eps=1e-6))),source=(('checkpoint/component','fill-identity','layout-identity'),),input_layout=layout,rotary_layout=(((1,1797,1,48),(86256,48,48,1),'torch.float32',0),)*2,slots=(N(metadata=((2048,2048),(2048,1),'torch.float32','cuda:0')),),attention=N(identity=('kitchen-int8-split/1',record['kitchen_artifact']['artifact'],record['kitchen_artifact']['key'],'binary-identity')),policy=policy)
choices=(('batch:3/tail:0',(3*1797*2048*4,q['workspace_bytes'])),('batch:2/tail:1',(3*1797*2048*4,queries[3]['workspace_bytes'])))
source=repo/'src/cozy_runtime/internal/execution/native_vae.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Call');run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run');assignment=next(n for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='semantic' for t in n.targets))
expression=compile(ast.Expression(assignment.value.args[0]),str(source),'eval')
env={'inventory':inventory,'algorithms':algorithms,'choices':choices,'self':N(dense_identity='actual-prepared-dense-contract')}
def encode(values):return phase.configuration(eval(expression,values))
encoded=encode(env);assert encoded is not None
# All three original mutable constituents independently explain the refusal.
assert phase.configuration(asdict(policy)) is None
assert phase.configuration(list(choices)) is None
assert phase.configuration(algorithms) is None
assert phase.configuration({'value':1}) is None and phase.configuration([1]) is None
changed=dict(env);changed['inventory']=copy.copy(inventory);changed['inventory'].policy=replace(policy,workspace_limit=policy.workspace_limit+256);assert encode(changed)!=encoded
changed=dict(env);changed['choices']=choices[:1];assert encode(changed)!=encoded
changed=dict(env);changed['algorithms']=copy.deepcopy(algorithms);changed['algorithms'][0][1]['id']+=1;assert encode(changed)!=encoded
changed=dict(env);changed['inventory']=copy.copy(inventory);changed['inventory'].source=(('different-source','fill-identity','layout-identity'),);assert encode(changed)!=encoded
assert not torch.cuda.is_initialized()
result={'passed':True,'scope':'pure CPU producer semantic expression; actual GPU26075 query metadata plus declared scalar/layout values; no GPU execution','query_count':len(queries),'original_mutable_policy_menu_algorithms_rejected':True,'immutable_expression_serializes':True,'policy_choice_algorithm_source_changes_affect_key':True,'global_dict_list_rejection_preserved':True,'configuration_sha256':hashlib.sha256(encoded.encode()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'gpu_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'cuda_initialized':False}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

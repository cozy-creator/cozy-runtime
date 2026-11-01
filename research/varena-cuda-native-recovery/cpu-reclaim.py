import json,sys
from pathlib import Path
import torch,tensorfs
from cozy_runtime.internal.residency import Declared,Memory,Weights

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
assert not torch.cuda.is_initialized()
writer=tensorfs.Writer(root/'weights.cozytensors');component=writer.component('small')
for layer,scale,bias in [('0',.25,0.),('1',.5,1.)]:
    component.put(layer+'.weight',torch.full((4,4),scale))
    component.put(layer+'.bias',torch.full((4,),bias))
source=writer.commit()
network=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,4))
vars(network)['_cozy_weights']=Declared(Weights(source,'small'),blocks=['0','1'])
memory=Memory('cpu');model=memory.load(network,'ram')
with torch.no_grad(),model.passing('cpu-reclaim-before',executing=True):
    before=network(torch.ones((1,4))).clone()
retained_before=[block.data is not None for block in model.groups()]
generations_before=[block.generation for block in model.groups()]
reclaimed=memory.reclaim()
retained_after=[block.data is not None for block in model.groups()]
generations_after=[block.generation for block in model.groups()]
with torch.no_grad(),model.passing('cpu-reclaim-after',executing=True):
    after=network(torch.ones((1,4))).clone()
assert torch.equal(before,after) and torch.equal(after,torch.full((1,4),3.))
assert any(retained_before) and not any(retained_after)
model.close();source.close()
assert not torch.cuda.is_initialized()
print(json.dumps({'passed':True,'device':'cpu','before':before.tolist(),'after':after.tolist(),
 'retained_before':retained_before,'retained_after':retained_after,
 'generations_before':generations_before,'generations_after':generations_after,
 'reported_reclaimed_bytes':reclaimed,'models_after_close':len(memory.models),'cuda_initialized':False}))

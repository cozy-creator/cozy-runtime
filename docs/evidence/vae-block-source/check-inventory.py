from pathlib import Path
import hashlib
import json
import sys

sys.path[:0]=[
 '/home/fidika/cozy_v2/packages/minimax-h3-tools/src',
 '/home/fidika/cozy/.worktrees/cozy-runtime/xs138-p2-vae-block/src',
 '/home/fidika/cozy/.worktrees/packages/xs138-p2-h3-output/minimax-h3',
 '/home/fidika/cozy_v2/outputs/xs138-p2-native-retention/metadata',
 '/home/fidika/cozy_v2/outputs/xs138-native-cache-candidate-a96c7c3e8/installed',
 '/home/fidika/cozy/.worktrees/cozy-creator/xs138-p0-torch/.task/torch-sdk/2.14.0+cu130-08187d9e/site-packages',
]
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoTransformerBlock
from cozy_runtime import fusion
from minimax_h3.model import VAE_BLOCK

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
with torch.device('meta'):
 block=MiniMaxH3VideoTransformerBlock(dim=2048,heads=32,dim_head=64)
block.eval().requires_grad_(False)
site=fusion._site(block,VAE_BLOCK,fusion._Switch(),'fixture.block')
linears=(*site.qkv,site.out,*site.inputs,site.down)
assert len(linears)==6 and all(type(m) is torch.nn.Linear for m in linears)
assert [(m.out_features,m.in_features) for m in linears]==[(2048,2048)]*4+[(16384,2048),(2048,8192)]
assert all(m.bias is not None and m.bias.numel()==m.out_features for m in linears)
assert all(not m.weight.requires_grad and not m.bias.requires_grad for m in linears)
assert site.qk[0].weight is None and site.qk[1].weight is None
assert VAE_BLOCK.rows=='fp16' and VAE_BLOCK.rotary==48 and VAE_BLOCK.mlp.value_first
matrices=[{'shape':list(m.weight.shape),'weight_dtype':str(m.weight.dtype),'bias_shape':list(m.bias.shape),'bias_dtype':str(m.bias.dtype),'fp16_cast_bytes':m.weight.numel()*2} for m in linears]
assert max(m['fp16_cast_bytes'] for m in matrices)==67108864
assert sum(m['fp16_cast_bytes'] for m in matrices)==134217728
geometries=[]
for capacity,tail in ((1,1),(2,1),(3,2),(7,1),(14,13),(28,14),(28,27),(28,28)):
 with torch.device('meta'):
  a=torch.empty(capacity*1797,16384,dtype=torch.float16)
  b=torch.empty(capacity*1797,8192,dtype=torch.float16)
  output=torch.empty(28,1797,2048,dtype=torch.float32)
 count=capacity*1797*2048
 av=a.view(-1);bv=b.view(-1)
 # Five original disjoint Q/K/V/Qnorm/Knorm regions; later raw/normalized
 # lifetimes may reuse these regions without changing any tensor pitch.
 slots=[av.narrow(0,i*count,count).view(capacity,1797,2048)[:tail] for i in range(5)]
 assert all(x.is_contiguous() and x.storage_offset()*x.element_size()%256==0 for x in slots)
 assert all(x.unflatten(-1,(32,64)).transpose(1,2).stride()==(1797*2048,64,2048,1) for x in slots)
 bhnd=slots[0].view(tail,32,1797,64)
 nhd=slots[1]
 assert bhnd.is_contiguous() and nhd.is_contiguous()
 up=av[:tail*1797*16384].view(tail*1797,16384)
 value,gate=up.chunk(2,dim=-1)
 activated=bv[:tail*1797*8192].view(tail*1797,8192)
 normed=bv[:tail*1797*2048].view(tail*1797,2048)
 down=av[:tail*1797*2048].view(tail*1797,2048)
 assert gate.stride()==value.stride()==(16384,1)
 assert all(x.is_contiguous() for x in (up,activated,normed,down))
 assert all(x.storage_offset()*x.element_size()%256==0 for x in (up,activated,normed,down))
 target=output[:tail]
 assert target.is_contiguous() and target.stride()==(1797*2048,2048,1)
 kitchen=[('q',(capacity,32,1797,64),1),('k',(capacity,32,1797,64),1),('v',(capacity*32*64,1856),1),('q_scale',(capacity,32,15*32),4),('k_scale',(capacity,32,29*4),4),('v_scale',(capacity*32*64,),4),('anchor',(capacity,32),4)]
 sizes={name:__import__('math').prod(shape)*width for name,shape,width in kitchen}
 geometries.append({'window':capacity,'tail':tail,'a_bytes':a.numel()*2,'b_bytes':b.numel()*2,'full28_output_bytes':output.numel()*4,'kitchen_bytes':sizes,'float_qkv_input_stride':list(slots[0].unflatten(-1,(32,64)).transpose(1,2).stride()),'attention_output_stride':list(bhnd.stride()),'ragged_packed_up_stride':list(up.stride()),'ragged_gate_stride':list(gate.stride()),'all_dense_and_kitchen_output_views_contiguous':True,'matrix_slot_byte_offsets_multiple_of256':True})
assert not torch.cuda.is_initialized()
result={'passed':True,'torch':str(torch.__version__),'torch_git':torch.version.git_version,'cuda_initialized':False,'actual_block_type':type(block).__name__,'actual_fused_site':True,'six_biased_plain_dense_modules':matrices,'layout_contract':VAE_BLOCK.contract(),'kernel_identity':VAE_BLOCK.kernels.identity,'geometries':geometries,'scope':'actual meta-device model construction and full/ragged tensor-layout inventory only; no CUDA query, execution, allocation-census, numeric or performance proof','source_sha256':{'fusion':hashlib.sha256(Path(fusion.__file__).read_bytes()).hexdigest(),'package_model':hashlib.sha256(Path(sys.modules['minimax_h3.model'].__file__).read_bytes()).hexdigest()}}
path=Path(sys.argv[1]); assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('layout_contract','geometries','six_biased_plain_dense_modules')}))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

## -- utils -- ##
def _num_groups(c: int) -> int:
    for g in [8,4,2,1]:
        if c % g == 0:
            return g
    return 1

class DropPath(nn.Module):
    def __init__(self, p:float=0.0): super().__init__(); self.p=float(p)
    def forward(self,x):
        if self.p==0.0 or not self.training: return x
        keep=1.0-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1)
        return x*(torch.rand(shape,device=x.device,dtype=x.dtype)<keep).float()/keep
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim:int): super().__init__(); self.dim=dim
    def forward(self,t:torch.Tensor)->torch.Tensor:
        half=self.dim//2
        freqs=torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)*-(math.log(10000.0)/(half-1)))
        a=t[:,None]*freqs[None,:]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

def sinusoidal_position_embedding_2d(h:int, w:int, dim:int, device=None, dtype=None) -> torch.Tensor:
    """
    2D sinuousidal PE:
        output shape = (1,h*w, dim)
    """
    # compute with fp 32. and cast at the end
    work_dtype = torch.float32

    half = dim // 2
    q = half // 2
    
    y = torch.arange(h, device=device, dtype=work_dtype).view(h,1)
    x = torch.arange(w, device=device, dtype=work_dtype).view(w,1)

    div_y = math.log(10000.0) / max(q - 1, 1)
    div_x = math.log(10000.0) / max(q - 1, 1)
    dv_y = torch.exp(torch.arange(q, device=device, dtype=work_dtype) * -div_y)
    dv_x = torch.exp(torch.arange(q, device=device, dtype=work_dtype) * -div_x)

    y_term = y * dv_y.view(1, q)
    x_term = x * dv_x.view(1, q)

    pe = torch.zeros(h,w,dim,device=device,dtype=work_dtype)

    # x part
    pe[:, :, 0:q]       = torch.sin(y_term)[:, None, :].expand(h, w, q)
    pe[:, :, q:half]    = torch.cos(y_term)[:, None, :].expand(h, w, q)

    # y part
    pe[:, :, half:half+q]   = torch.sin(x_term)[None, :, :].expand(h, w, q)
    pe[:, :, half+q:dim]    = torch.cos(x_term)[None, :, :].expand(h, w, q)

    return pe.to(dtype).view(1, h*w, dim)

class ChannelAffine(nn.Module):
    def __init__(self,ch:int): super().__init__(); self.scale=nn.Parameter(torch.ones(ch)); self.bias=nn.Parameter(torch.zeros(ch))
    def forward(self,x): return x*self.scale.view(1,-1,1,1)+self.bias.view(1,-1,1,1)


class ConvBlockGN(nn.Module):
    def __init__(self,c_in,c_out,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(c_in,c_out,3,stride=stride,padding=1,bias=False)
        self.gn=nn.GroupNorm(_num_groups(c_out),c_out)
        self.dw=nn.Conv2d(c_out,c_out,3,1,1,bias=False,groups=c_out)
        self.gn2=nn.GroupNorm(_num_groups(c_out),c_out)
        self.act=nn.GELU()
    def forward(self,x):
        x=self.act(self.gn(self.conv(x)))
        x=self.act(self.gn2(self.dw(x)))
        return x
    

class StemGN(nn.Module):
    def __init__(self,in_ch,mid_ch):
        super().__init__()
        self.s2a=ConvBlockGN(in_ch,mid_ch,2) #128->64
        self.s2b=ConvBlockGN(mid_ch,mid_ch,2)#64->32
        self.s2c=ConvBlockGN(mid_ch,mid_ch,2)#32->16
    def forward_32(self,x): return self.s2b(self.s2a(x))
    def forward_16(self,x): return self.s2c(self.s2b(self.s2a(x)))



class MultiheadLinearAttention(nn.Module):
    def __init__(self, dim:int, heads:int=8, dropout:float=0.0, attn_kernel:str="rff", rff_dim:int=128):
        super().__init__()
        self.dim, self.heads, self.dk = dim, heads, dim//heads
        self.to_q=nn.Linear(dim,dim,bias=False); self.to_k=nn.Linear(dim,dim,bias=False); self.to_v=nn.Linear(dim,dim,bias=False)
        self.proj=nn.Linear(dim,dim); self.drop=nn.Dropout(dropout)
        self.kernel=attn_kernel; self.rff_dim=rff_dim
        self.register_buffer("W", torch.randn(heads,self.dk,rff_dim)/math.sqrt(self.dk))
        self.register_buffer("b", 2*math.pi*torch.rand(heads,rff_dim))

    @staticmethod
    def phi_elu(x): 
        return F.elu(x, inplace=False)+1.0
    
    def phi_rff(self,x):
        B,N,H,dk=x.shape
        proj=torch.einsum('bnhd,hdr->bnhr',x,self.W)+self.b
        feat=torch.cat([torch.cos(proj),torch.sin(proj)],dim=-1)/math.sqrt(self.rff_dim)
        return feat
    
    def _apply_feature_map(self,x): 
        return self.phi_rff(x) if self.kernel=="rff" else self.phi_elu(x)
    
    def forward(self, x, rope=None):
        B, N, D = x.shape
        H, dk = self.heads, self.dk
        q = self.to_q(x).view(B, N, H, dk)
        k = self.to_k(x).view(B, N, H, dk)
        v = self.to_v(x).view(B, N, H, dk)

        if rope is not None:
            def apply_rope(t):
                d = t.shape[-1]
                rope_d = rope[..., :d].to(t.dtype)
                half = d // 2
                sin = rope_d[:, :half][None, :, None, :]
                cos = rope_d[:, half:][None, :, None, :]
                t1, t2 = t[..., :half], t[..., half:]

                return torch.cat([t1 * cos - t2 * sin, 
                                t1 * sin + t2 * cos], dim=-1)# (B, N, H, half) -> (B, N, H, d)

            q = apply_rope(q)
            k = apply_rope(k)
        q=self._apply_feature_map(q); k=self._apply_feature_map(k)
        Kv=torch.einsum('bnhf,bnhd->bhfd',k,v)
        z=1.0/(torch.einsum('bnhf,bhf->bnh',q,k.sum(dim=1))+1e-6)
        out=torch.einsum('bnhf,bhfd->bnhd',q,Kv)*z.unsqueeze(-1)
        out=out.contiguous().view(B,N,D)
        return self.drop(self.proj(out))


class PerformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.0, drop_path=0.1, attn_kernel="rff", rff_dim=128):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=MultiheadLinearAttention(dim,heads,dropout,attn_kernel,rff_dim); self.dp1=DropPath(drop_path)
        self.n2=nn.LayerNorm(dim); hidden=int(dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,dim),nn.Dropout(dropout))
        self.dp2=DropPath(drop_path)
    def forward(self,x,rope=None):
        x=x+self.dp1(self.attn(self.n1(x),rope))
        x=x+self.dp2(self.mlp(self.n2(x)))
        return x
    


class OccPerformerEncoder_v2(nn.Module):
    def __init__(self,in_ch=4,in_hw=128,emb_dim=256,heads=8,layers=6,dropout=0.0,drop_path=0.1,
                 attn_kernel="rff",rff_dim=128,patch32=4,patch16=2,use_rope=True,use_timestep_film=True,return_tokens=True):
        super().__init__()
        self.in_ch=in_ch; self.in_hw=in_hw; self.emb_dim=emb_dim; self.return_tokens=return_tokens
        self.use_rope=use_rope; self.use_timestep_film=use_timestep_film
        self.chan_aff=ChannelAffine(in_ch)
        stem_ch=emb_dim//2; self.stem=StemGN(in_ch,stem_ch)
        self.proj32=nn.Linear(stem_ch*patch32*patch32,emb_dim)
        self.proj16=nn.Linear(stem_ch*patch16*patch16,emb_dim)
        self.cls=nn.Parameter(torch.zeros(1,1,emb_dim))
        dpr=[x.item() for x in torch.linspace(0,drop_path,layers)]
        self.blocks=nn.ModuleList([PerformerBlock(emb_dim,heads,4.0,dropout,dpr[i],attn_kernel,rff_dim) for i in range(layers)])
        self.post_ln=nn.LayerNorm(emb_dim)
        if self.use_timestep_film:
            self.t_sin=SinusoidalPosEmb(emb_dim)
            self.t_film=nn.Linear(emb_dim, emb_dim*2)

    def _patchify(self,x,patch:int):
        B,C,H,W=x.shape; assert H%patch==0 and W%patch==0
        unf=F.unfold(x,kernel_size=(patch,patch),stride=(patch,patch))
        toks=unf.transpose(1,2); N=toks.shape[1]
        return toks, H//patch, W//patch

    def _rope_cache(self,N:int,dim:int,device,dtype):
        half=dim//2; pos=torch.arange(N,device=device,dtype=dtype)
        freqs=torch.exp(torch.arange(half,device=device,dtype=dtype)*-(math.log(10000.0)/(half-1)))
        ang=pos[:,None]*freqs[None,:]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, grid:torch.Tensor, timesteps:Optional[torch.Tensor]=None):
        B,C,H,W=grid.shape; assert C==self.in_ch and H==self.in_hw and W==self.in_hw
        x=self.chan_aff(grid)
        x32=self.stem.forward_32(x); x16=self.stem.forward_16(x)
        tok32, th32, tw32=self._patchify(x32,4); tok16, th16, tw16=self._patchify(x16,2)
        tok32=self.proj32(tok32); tok16=self.proj16(tok16)
        tok32=tok32+sinusoidal_position_embedding_2d(th32,tw32,self.emb_dim,device=tok32.device,dtype=tok32.dtype)
        tok16=tok16+sinusoidal_position_embedding_2d(th16,tw16,self.emb_dim,device=tok16.device,dtype=tok16.dtype)
        tokens=torch.cat([tok32,tok16],dim=1)
        cls=self.cls.expand(B,-1,-1)
        if self.use_timestep_film and timesteps is not None:
            if not torch.is_tensor(timesteps): timesteps=torch.tensor([timesteps],device=grid.device,dtype=torch.long)
            if timesteps.ndim==0: timesteps=timesteps[None]
            tfeat=self.t_sin(timesteps); gamma,beta=self.t_film(tfeat).chunk(2,dim=-1)
            cls=(1.0+gamma).unsqueeze(1)*cls + beta.unsqueeze(1)
        xseq=torch.cat([cls,tokens],dim=1)
        dk = self.emb_dim // self.blocks[0].attn.heads
        rope = self._rope_cache(xseq.size(1), dk, xseq.device, xseq.dtype)
        for blk in self.blocks: xseq=blk(xseq, rope=rope)
        xseq=self.post_ln(xseq)
        z_map=xseq[:,0,:]; toks=xseq[:,1:,:] if self.return_tokens else None
        return z_map, toks







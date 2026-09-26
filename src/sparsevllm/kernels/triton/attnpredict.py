"""AttentionPredictor history ring and shared-token views."""
import triton as tr
import triton.language as tl


@tr.jit
def seed(History, Maxima, Norm, Rows, Lens, Chunks, Cursor, N:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);t=tl.program_id(2)
    row=tl.load(Rows+b);take=tl.minimum(tl.load(Chunks+b),64);i=tl.arange(0,W)
    src=t-(64-take)
    z=tl.load(Maxima+((b*H+h)*64+src)*N+i,(i<N)&(src>=0),-float('inf'))
    norm=tl.load(Norm+(b*H+h)*64+src,src>=0,0)
    p=tl.exp(z-norm).to(tl.bfloat16).to(tl.float16)
    tl.store(History+((row*H+h)*64+t)*N+i,p,i<N)
    if h==0 and t==0:tl.store(Cursor+row,0)


@tr.jit
def feedback(Scores,Positions,Lens,Pool,Rows,Writes,N:tl.constexpr,H:tl.constexpr,P:tl.constexpr,W:tl.constexpr,SCALE:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);i=tl.arange(0,W)
    if tl.load(Writes+b)>=0:
        length=tl.load(Lens+b)
        z=tl.load(Scores+(b*H+h)*P+i,i<length,-float('inf'))*SCALE
        p=tl.exp(z-tl.max(z,0));p=(p/tl.sum(p,0)).to(tl.bfloat16).to(tl.float32)
        block=tl.load(Positions+b*P+i,i<length,0)//16
        # Pool into FP32 workspace; FP16 atomic_max is not portable.
        tl.atomic_max(Pool+(b*H+h)*N+block,p,i<length,sem='relaxed')


@tr.jit
def append(Pool,History,Cursor,Rows,Writes,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);i=tl.program_id(2)*W+tl.arange(0,W)
    if tl.load(Writes+b)>=0:
        row=tl.load(Rows+b);t=tl.load(Cursor+row)
        p=tl.load(Pool+(b*H+h)*N+i,i<N,0)
        tl.store(History+((row*H+h)*64+t)*N+i,p,i<N)


@tr.jit
def advance(Cursor,Rows,Writes,B:tl.constexpr,W:tl.constexpr):
    i=tl.arange(0,W);row=tl.load(Rows+i,i<B,0);valid=(i<B)&(tl.load(Writes+i,i<B,-1)>=0)
    t=tl.load(Cursor+row,valid,0);tl.store(Cursor+row,(t+1)%64,valid)


@tr.jit
def pack(History,Cursor,Rows,Lens,Input,Valid,N:tl.constexpr,H:tl.constexpr,C:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);t=tl.program_id(2);i=tl.arange(0,W)
    row=tl.load(Rows+b);cursor=tl.load(Cursor+row);end=tl.maximum(tl.cdiv(tl.load(Lens+b),16)-8,0)
    x=tl.load(History+((row*H+h)*64+(t+cursor)%64)*N+i+4,(i<C)&(i<end),0)
    tl.store(Input+((b*H+h)*64+t)*C+i,x,i<C)
    if t==0:tl.store(Valid+(b*H+h)*C+i,i<end,i<C)


@tr.jit
def select_positions(Top,Rows,Lens,Writes,Keep,Counts,K:tl.constexpr,P:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);row=tl.load(Rows+b);n=tl.load(Lens+b);i=tl.arange(0,W)
    if tl.load(Writes+b)>=0:
        candidates=tl.maximum(tl.cdiv(n,16)-8,0);k=tl.minimum(K,candidates)
        j=(i-64)//16
        blk=tl.load(Top+b*K+j,(j>=0)&(j<k),0)+4
        pos=tl.where(i<64,i,tl.where(i<64+k*16,blk*16+(i-64)%16,n-64+i-64-k*16))
        valid=(i<128+k*16)&(pos>=0)&(pos<n)
        pos=tl.sort(tl.where(valid,pos,2147483647),descending=False)
        prev=tl.gather(pos,tl.maximum(i-1,0),0)
        unique=(pos<2147483647)&((i==0)|(pos!=prev))
        dest=tl.cumsum(unique.to(tl.int32),0)-1
        tl.store(Keep+row*P+dest,pos,unique)
        tl.store(Counts+row,tl.sum(unique.to(tl.int32),0))


@tr.jit
def view(Keep,Counts,Rows,Lens,Writes,Table,Positions,Slots,OutLens,N:tl.constexpr,P:tl.constexpr,W:tl.constexpr,APPEND_CURRENT:tl.constexpr=True):
    b=tl.program_id(0);row=tl.load(Rows+b);i=tl.arange(0,W);n=tl.load(Lens+b)
    count=tl.load(Counts+row);valid=tl.load(Writes+b)>=0
    pos=tl.load(Keep+row*(P-1)+i,i<count,0)
    if APPEND_CURRENT:pos=tl.where(i==count,n-1,pos)
    count=count+int(APPEND_CURRENT)
    slot=tl.load(Table+row*N+pos,(i<count)&valid,0)
    tl.store(Positions+b*P+i,pos,i<P);tl.store(Slots+b*P+i,slot,i<P)
    tl.store(OutLens+b,tl.where(valid,count,0))

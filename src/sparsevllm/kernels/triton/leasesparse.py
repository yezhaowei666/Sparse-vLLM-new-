"""Lease state transitions, block EMA, and indexed shared-pool KV recall."""
import triton as tr
import triton.language as tl


@tr.jit
def begin(Starts, PendingStarts, Pending, Bank, Refresh, Rows, Lengths, Writes,
          R: tl.constexpr, B: tl.constexpr, T: tl.constexpr, REUSE: tl.constexpr = 16, PREDICT: tl.constexpr = -1):
    i = tl.arange(0, B)
    row = tl.load(Rows+i, i<T, 0)
    valid = (i<T) & (tl.load(Writes+i, i<T, -1)>=0)
    n = tl.load(Lengths+i, valid, 0)
    p = tl.load(Pending+row, valid, 0)
    start = tl.load(Starts+row, valid, 0)
    refresh = (tl.sum((valid & (n-start>=REUSE)).to(tl.int32), 0)>0) & (tl.sum(p,0)==0)
    if REUSE == 1:
        refresh = tl.sum(valid.to(tl.int32), 0)>0
    if PREDICT >= 0: refresh = tl.full((), PREDICT, tl.int32)
    tl.store(Refresh, refresh.to(tl.int32))
    next_start = tl.load(PendingStarts+row, valid, 0)
    bank = tl.load(Bank+row, valid, 0)
    tl.store(Starts+row, tl.where(p!=0,next_start,start),valid)
    tl.store(Bank+row, tl.where(p!=0,1-bank,bank),valid)
    tl.store(Pending+row,0,valid)


@tr.jit
def view(Hot, Bank, Starts, PendingStarts, Rows, Lengths, Writes, Table, Positions, Slots, Lens,
         MAXN:tl.constexpr, P:tl.constexpr, K:tl.constexpr, NEXT:tl.constexpr,
         Refresh, W:tl.constexpr):
    b=tl.program_id(0); row=tl.load(Rows+b); n=tl.load(Lengths+b)
    valid=tl.load(Writes+b)>=0
    if not NEXT or tl.load(Refresh)!=0:
        bank=tl.load(Bank+row)
        if NEXT: bank=1-bank
        j=tl.arange(0,W)
        block=tl.load(Hot+(row*2+bank)*K+j,j<K,-1)
        limit=tl.load(PendingStarts+row) if NEXT else tl.load(Starts+row)
        count=tl.sum(tl.where(block>=0,tl.minimum(16,tl.maximum(tl.minimum(n,limit)-64-block*16,0)),0),0)
        sink=tl.minimum(n,64); recent=tl.minimum(64,tl.maximum(n-sink,0))
        length=tl.minimum(n,sink+count+recent)
        # Short contexts retain every token; otherwise hot blocks are sorted.
        h=(j-sink)//16
        blk=tl.load(Hot+(row*2+bank)*K+h,(h>=0)&(h<K),0)
        pos=tl.where(j<sink,j,tl.where(j<sink+count,blk*16+(j-sink)%16,n-recent+j-sink-count))
        pos=tl.where(n<=128,j,pos)
        slot=tl.load(Table+row*MAXN+pos,(j<length)&valid,0)
        tl.store(Positions+b*P+j,pos,j<P)
        tl.store(Slots+b*P+j,slot,j<P)
        tl.store(Lens+b,tl.where(valid,length,0))


@tr.jit
def clear_pool(Pool, Observed, Refresh, N:tl.constexpr, H:tl.constexpr, W:tl.constexpr):
    if tl.load(Refresh)!=0:
        i=tl.program_id(0)*W+tl.arange(0,W)
        tl.store(Pool+i,0,i<N*H)
        tl.store(Observed+i,0,i<N)


@tr.jit
def pool_decode(Scores, Positions, Lens, Pool, Observed, Refresh,
                P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,W:tl.constexpr,SCALE:tl.constexpr):
    if tl.load(Refresh)!=0:
        b=tl.program_id(0); h=tl.program_id(1); i=tl.arange(0,W)
        length=tl.load(Lens+b)
        logits=tl.load(Scores+b*S0+h*S1+i,i<length,-float('inf'))*SCALE
        prob=tl.exp(logits-tl.max(logits,0)); prob=prob/tl.sum(prob,0)
        pos=tl.load(Positions+b*P+i,i<length,0)//16
        tl.atomic_max(Pool+(b*H+h)*N+pos,prob,i<length,sem='relaxed')
        if h==0: tl.atomic_or(Observed+b*N+pos,1,i<length,sem='relaxed')


@tr.jit
def ema(State, Seen, Pool, Observed, Rows, Refresh,
        N:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    if tl.load(Refresh)!=0:
        b=tl.program_id(0); h=tl.program_id(1); t=tl.program_id(2)*W+tl.arange(0,W)
        row=tl.load(Rows+b); observed=tl.load(Observed+b*N+t,t<N,0)!=0
        seen=tl.load(Seen+row*N+t,t<N,0)!=0
        old=tl.load(State+(row*H+h)*N+t,t<N,0)
        score=tl.load(Pool+(b*H+h)*N+t,t<N,0)
        out=tl.where(seen,old*0.8+score*0.2,score)
        tl.store(State+(row*H+h)*N+t,out,(t<N)&observed)
        # Seen is published in a separate kernel after every head consumed it.


@tr.jit
def publish_seen(Seen, Observed, Rows, Refresh,N:tl.constexpr,W:tl.constexpr):
    if tl.load(Refresh)!=0:
        b=tl.program_id(0); t=tl.program_id(1)*W+tl.arange(0,W); row=tl.load(Rows+b)
        observed=tl.load(Observed+b*N+t,t<N,0)
        tl.store(Seen+row*N+t,1,(t<N)&(observed!=0))


@tr.jit
def reduce_heads(State, Reduced, Rows, Refresh,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,HH:tl.constexpr):
    if tl.load(Refresh)!=0:
        b=tl.program_id(0); row=tl.load(Rows+b)
        n=tl.program_id(1)*W+tl.arange(0,W); h=tl.arange(0,HH)
        x=tl.load(State+(row*H+h[:,None])*N+n[None,:],(h[:,None]<H)&(n[None,:]<N),0)
        tl.store(Reduced+b*N+n,tl.max(x,0),n<N)


@tr.jit(do_not_specialize=["N"])
def select(Reduced, Hot, Bank, Pending, PendingStarts, Rows, Lengths, Writes, Refresh, Counts,
           N,K:tl.constexpr,W:tl.constexpr,INIT:tl.constexpr):
    if tl.load(Refresh)!=0:
        b=tl.program_id(0); row=tl.load(Rows+b); n=tl.load(Lengths+b)
        if tl.load(Writes+b)>=0:
            TOP:tl.constexpr=tr.next_power_of_2(K)
            i=tl.arange(0,max(W,TOP))
            end=tl.cdiv(n,16)-4
            valid=(i>=4)&(i<end)&(i<N)
            x=tl.load(Reduced+b*N+i,i<N,0)
            # Nonnegative probability scores; ties use ascending block position.
            key=(x.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)|(0xffffffff-i.to(tl.uint64))
            key=tl.where(valid,key,0)
            # Keep only the top candidates; sorting the entire history twice
            # creates a large unrolled network and expensive CUDA compilation.
            sorted_key=tl.topk(key,TOP)
            count=tl.minimum(K,tl.maximum(end-4,0))
            j=tl.arange(0,TOP)
            ids=0xffffffff-(sorted_key&0xffffffff)
            ordered=tl.sort(tl.where(j<count,ids,0xffffffff),descending=False)
            bank=1-tl.load(Bank+row)
            tl.store(Hot+(row*2+bank)*K+j,tl.where(j<count,ordered.to(tl.int32),-1),j<K)
            tl.store(Pending+row,1);tl.store(PendingStarts+row,n)
            tl.atomic_add(Counts+int(INIT),1)


@tr.jit
def tail_logits(Q, Keys, Table, Rows, Lens, Starts, Chunks, Maxima, Lses,
                N:tl.constexpr,MAXN:tl.constexpr,H:tl.constexpr,KH:tl.constexpr,D:tl.constexpr,
                QS:tl.constexpr,QH:tl.constexpr,KS:tl.constexpr,KHS:tl.constexpr,SCALE:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);tile=tl.program_id(2)
    n=tl.load(Lens+b);chunk=tl.load(Chunks+b);take=tl.minimum(chunk,64)
    row=tl.load(Rows+b);start=tl.load(Starts+b)+chunk-take
    qi=tl.arange(0,64);ki=tile*128+tl.arange(0,128);d=tl.arange(0,D)
    q=tl.load(Q+(start+qi[:,None])*QS+h*QH+d[None,:],qi[:,None]<take,0)
    slot=tl.load(Table+row*MAXN+ki,ki<n,0)
    k=tl.load(Keys+slot[None,:]*KS+(h//(H//KH))*KHS+d[:,None],ki[None,:]<n,0)
    z=tl.dot(q,k)*SCALE
    z=tl.where((ki[None,:]<=n-take+qi[:,None])&(ki[None,:]<n)&(qi[:,None]<take),z,-float('inf'))
    z=tl.reshape(z,(64,8,16));m=tl.max(z,2)
    total=tl.sum(tl.exp(z-m[:,:,None]),2)
    lse=tl.where(m>-float('inf'),m+tl.log(total),-float('inf'))
    blocks=tile*8+tl.arange(0,8)
    offsets=((b*H+h)*64+qi[:,None])*N+blocks[None,:]
    tl.store(Maxima+offsets,m,blocks[None,:]<N)
    tl.store(Lses+offsets,lse,blocks[None,:]<N)


@tr.jit
def tail_norm(Lses, Norm,N:tl.constexpr,W:tl.constexpr):
    t=tl.program_id(0);i=tl.arange(0,W);z=tl.load(Lses+t*N+i,i<N,-float('inf'))
    m=tl.max(z,0);den=m+tl.log(tl.sum(tl.exp(z-m),0));tl.store(Norm+t,den)


@tr.jit
def tail_ema(State, Seen, Maxima, Norm, Rows, Lens, Chunks,
             N:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);i=tl.program_id(2)*W+tl.arange(0,W)
    row=tl.load(Rows+b);n=tl.load(Lens+b);take=tl.minimum(tl.load(Chunks+b),64)
    state=tl.full((W,),0,tl.float32);seen=tl.full((W,),False,tl.int1)
    for t in range(take):
        z=tl.load(Maxima+((b*H+h)*64+t)*N+i,i<N,-float('inf'))
        norm=tl.load(Norm+(b*H+h)*64+t)
        score=tl.exp(z-norm);observed=(i*16<=n-take+t)&(i<N)
        state=tl.where(observed,tl.where(seen,state*0.8+score*0.2,score),state)
        seen=seen|observed
    tl.store(State+(row*H+h)*N+i,state,i<N)
    if h==0: tl.store(Seen+row*N+i,seen.to(tl.int32),i<N)


@tr.jit
def plan_pool(Keys, Directory, Table, Rows, Lengths, Writes, Positions, ViewLens,
              OutSlots, Free, Refresh,
              HOSTS:tl.constexpr, MAXN:tl.constexpr, P:tl.constexpr, NEXT:tl.constexpr):
    b=tl.program_id(0)
    if (not NEXT or tl.load(Refresh)!=0) and tl.load(Writes+b)>=0:
        row=tl.load(Rows+b);n=tl.load(Lengths+b);length=tl.load(ViewLens+b)
        i=tl.arange(0,P)
        pos=tl.load(Positions+b*P+i,i<length,0)
        valid=i<length
        host=tl.load(Table+row*MAXN+pos,valid,0)
        cached=tl.load(Directory+row*HOSTS+host,valid,-1)
        hit=valid&(cached>=0)&(tl.load(Keys+cached,cached>=0,-1)==host)
        missing=valid&~hit
        target=cached
        if tl.sum(missing.to(tl.int32),0)>0:
            empty=tl.load(Keys+row*P+i)<0
            vacant=tl.min(tl.where(empty,i,P),0)
            if not NEXT and tl.sum(missing.to(tl.int32),0)==1 and vacant<P:
                allocated=tl.full((P,),row*P+vacant,tl.int32)
            else:
                # The layer has finished reading; protect only the new selection, then reuses scratch as a free-slot list.
                tl.store(Free+b*P+i,0)
                tl.debug_barrier()
                tl.atomic_or(Free+b*P+cached-row*P,1,hit,sem="relaxed")
                tl.debug_barrier()
                protected=tl.load(Free+b*P+i)!=0
                unused=~protected&empty
                victim=~protected&~empty
                rank=tl.where(unused,tl.cumsum(unused.to(tl.int32),0)-1,
                              tl.sum(unused.to(tl.int32),0)+tl.cumsum(victim.to(tl.int32),0)-1)
                tl.debug_barrier()
                tl.store(Free+b*P+rank,row*P+i,~protected)
                tl.debug_barrier()
                missing_rank=tl.cumsum(missing.to(tl.int32),0)-1
                allocated=tl.load(Free+b*P+missing_rank,missing,0)
            target=tl.where(missing,allocated,cached)
            evicted=tl.load(Keys+target,missing,-1)
            tl.store(Directory+row*HOSTS+evicted,-1,missing&(evicted>=0))
            tl.store(Directory+row*HOSTS+host,target,missing)
        tl.store(OutSlots+b*P+i,tl.where(valid,target,0),i<P)


@tr.jit
def recall(HostPtrs, GPU, Keys, Table, Rows, Lengths, Writes, Positions, ViewLens,
           OutSlots, Current, Refresh,
           MAXN:tl.constexpr,P:tl.constexpr,WIDTH:tl.constexpr,
           NEXT:tl.constexpr,COMP:tl.constexpr,W:tl.constexpr,CS:tl.constexpr,T:tl.constexpr=16):
    b=tl.program_id(0)
    if (not NEXT or tl.load(Refresh)!=0) and tl.load(Writes+b)>=0:
        n=tl.load(Lengths+b);row=tl.load(Rows+b);length=tl.load(ViewLens+b)
        for tile in range(tl.program_id(1),tl.cdiv(length,T),tl.num_programs(1)):
            i=tile*T+tl.arange(0,T)
            pos=tl.load(Positions+b*P+i,i<length,0)
            valid=i<length
            slot=tl.load(Table+row*MAXN+pos,valid,0)
            target=tl.load(OutSlots+b*P+i,valid,0)
            changed=valid&((tl.load(Keys+target,valid,-1)!=slot)|((not NEXT)&(pos==n-1)))
            if tl.sum(changed.to(tl.int32),0)>0:
                d=tl.arange(0,W)
                src=tl.load(HostPtrs+COMP).to(tl.pointer_type(GPU.dtype.element_ty))
                from_host=changed&((NEXT)|(pos!=n-1))
                value=tl.load(src+slot[:,None]*WIDTH+d[None,:],from_host[:,None]&(d[None,:]<WIDTH),0)
                if not NEXT:
                    cur=tl.load(Current+b*CS+d,d<WIDTH,0)
                    value=tl.where((pos==n-1)[:,None],cur[None,:],value)
                tl.store(GPU+target[:,None]*WIDTH+d[None,:],value,changed[:,None]&(d[None,:]<WIDTH))
            tl.debug_barrier()
            tl.store(Keys+target,slot,valid)

@tr.jit
def mark_use(Uses, Starts, Rows, Lengths, Writes, B:tl.constexpr):
    b=tl.program_id(0);row=tl.load(Rows+b)
    if tl.load(Writes+b)>=0:
        start=tl.load(Starts+row);old=tl.load(Uses+row*2)
        if start!=old:
            tl.store(Uses+row*2,start)
            tl.store(Uses+row*2+1,tl.load(Lengths+b)-1)


@tr.jit
def record_completion(Refresh, Counter, Completed):
    if tl.load(Refresh)!=0:
        serial=tl.atomic_add(Counter,1)+1
        tl.store(Completed,serial)


@tr.jit
def predictor_keys_prefill(K, Sums, Tail, Last, ROW, START, COUNT,
                           N:tl.constexpr, KH:tl.constexpr, D:tl.constexpr, KS:tl.constexpr, KHS:tl.constexpr):
    block=START//16+tl.program_id(0);h=tl.program_id(1)
    t=block*16+tl.arange(0,16);d=tl.arange(0,D)
    valid=(t>=START)&(t<START+COUNT)
    x=tl.load(K+(t[:,None]-START)*KS+h*KHS+d[None,:],valid[:,None],0).to(tl.float32)
    old=tl.load(Sums+((ROW*KH+h)*N+block)*D+d,block*16<START,0)
    tl.store(Sums+((ROW*KH+h)*N+block)*D+d,old+tl.sum(x,0))
    tl.store(Tail+((ROW*KH+h)*80+t[:,None]%80)*D+d[None,:],x,
             valid[:,None]&(t[:,None]>=START+COUNT-80))
    if block == (START+COUNT-1)//16: tl.store(Last+ROW*KH+h,START+COUNT-1)


@tr.jit
def predictor_keys_decode(K, Sums, Tail, Last, Rows, Lengths, Writes,
                          N:tl.constexpr, KH:tl.constexpr,D:tl.constexpr,KS:tl.constexpr,KHS:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1)
    if tl.load(Writes+b)>=0:
        row=tl.load(Rows+b);t=tl.load(Lengths+b)-1;d=tl.arange(0,D)
        x=tl.load(K+b*KS+h*KHS+d).to(tl.float32)
        ptr=Sums+((row*KH+h)*N+t//16)*D+d
        old=tl.load(ptr,t%16!=0,0)
        repeated=tl.load(Last+row*KH+h)==t
        previous=tl.load(Tail+((row*KH+h)*80+t%80)*D+d,repeated&(t%16!=0),0).to(tl.float32)
        tl.store(ptr,old+(x-previous))
        tl.store(Tail+((row*KH+h)*80+t%80)*D+d,x)
        tl.store(Last+row*KH+h,t)


@tr.jit
def predictor_queries(Q, History, Rows, Lengths, Writes, H:tl.constexpr,D:tl.constexpr,
                      QS:tl.constexpr,QH:tl.constexpr, PREFILL:tl.constexpr=False,
                      ROW=0, START=0, COUNT=0):
    b=tl.program_id(0);h=tl.program_id(1);i=tl.arange(0,8);d=tl.arange(0,D)
    if PREFILL:
        t=START+COUNT-8+i;row=ROW
        x=tl.load(Q+(t[:,None]-START)*QS+h*QH+d[None,:],t[:,None]>=START,0)
        tl.store(History+((row*H+h)*8+t[:,None]%8)*D+d[None,:],x,t[:,None]>=START)
    elif tl.load(Writes+b)>=0:
        row=tl.load(Rows+b);t=tl.load(Lengths+b)-1
        x=tl.load(Q+b*QS+h*QH+d)
        tl.store(History+((row*H+h)*8+t%8)*D+d,x)


@tr.jit
def predictor_pack(History,Tail,Rows,Lengths,Q,Partial,H:tl.constexpr,KH:tl.constexpr,D:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);row=tl.load(Rows+b);n=tl.load(Lengths+b)
    t=n-8+tl.arange(0,8);d=tl.arange(0,D)
    x=tl.load(History+((row*H+h)*8+t[:,None]%8)*D+d[None,:],t[:,None]>=0,0)
    tl.store(Q+((b*H+h)*8+tl.arange(0,8)[:,None])*D+d[None,:],x)
    if h<KH:
        cutoff=tl.maximum(n-64,0);count=(cutoff-1)%16+1
        tail_pos=(cutoff-1)//16*16+tl.arange(0,16)
        tail_values=tl.load(Tail+((row*KH+h)*80+tail_pos[:,None]%80)*D+d[None,:],
                  (tail_pos[:,None]>=0)&(tail_pos[:,None]<cutoff),0).to(tl.float32)
        tl.store(Partial+(b*KH+h)*D+d,tl.sum(tail_values,0)/count)


@tr.jit
def predictor_logits(Q,Sums,Partial,Rows,Lengths,Out,
                     N:tl.constexpr,C:tl.constexpr,H:tl.constexpr,KH:tl.constexpr,D:tl.constexpr,
                     QS:tl.constexpr,QF:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);tile=tl.program_id(2)
    row=tl.load(Rows+b);n=tl.load(Lengths+b);cutoff=n-64
    j=tl.arange(0,32);d=tl.arange(0,D);blocks=tile*32+tl.arange(0,32)
    heads=h*(H//KH)+j%(H//KH);future=j//(H//KH)
    q=tl.load(Q+b*QS+future[:,None]*QF+heads[:,None]*QH+d[None,:]*QD,future[:,None]<4,0)
    keys=tl.load(Sums+((row*KH+h)*N+blocks[None,:])*D+d[:,None],blocks[None,:]<N,0)/16
    last=(cutoff-1)//16;count=(cutoff-1)%16+1
    partial=tl.load(Partial+(b*KH+h)*D+d)
    keys=tl.where((blocks[None,:]==last)&(count!=16),partial[:,None],keys)
    z=tl.dot(q,keys,input_precision='tf32x3')*(D**-.5)
    z+=tl.where(blocks[None,:]==last,tl.log(count/16.),0)
    valid=(blocks>=4)&(blocks*16<cutoff)&(blocks<C)
    z=tl.where(valid[None,:],z,-float('inf'))
    tl.store(Out+((b*4+future[:,None])*H+heads[:,None])*C+blocks[None,:],z,
             (future[:,None]<4)&(blocks[None,:]<C))


@tr.jit
def predictor_probability(Scores,Mass,Lengths,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);i=tl.arange(0,W)
    z=tl.load(Scores+(b*4*H+h)*C+i,i<C,-float('inf'))
    if tl.load(Lengths+b)>128:
        p=tl.exp(z-tl.max(z,0));p=p/tl.sum(p,0)
        mass=tl.sigmoid(tl.load(Mass+b*4*H+h))
        tl.store(Scores+(b*4*H+h)*C+i,p*mass,i<C)
    else:
        tl.store(Scores+(b*4*H+h)*C+i,0,i<C)


@tr.jit
def predictor_reduce(Scores,Reduced,C:tl.constexpr,N:tl.constexpr,H:tl.constexpr):
    b=tl.program_id(0);i=tl.program_id(1)*32+tl.arange(0,32);h=tl.arange(0,128)
    x=tl.load(Scores+(b*4*H+h[:,None])*C+i[None,:],(h[:,None]<4*H)&(i[None,:]<C),0)
    tl.store(Reduced+b*N+i,tl.sum(x,0)/(4*H),i<C)

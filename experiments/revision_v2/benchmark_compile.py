"""Paired real-data benchmark, same initial weights/RNG/batch/optimizer."""
import argparse,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
from train_revision import load_base,configure_model
from performance_runtime import install

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--methods',nargs='+',default=['linear','lora','mobilenetv2'])
    a=p.parse_args()
    root=Path.cwd();manifest=json.loads((root/'reference_conservative_d8_v2.json').read_text())
    results=[]
    for method in a.methods:
        base=load_base(Path(__file__).with_name('train_tomato_full_base.py'))
        torch=base.torch;torch.set_num_threads(4)
        base.random.seed(42);base.np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False
        configure_model(base,method,8)
        mode='linear' if method=='linear' else 'lora'
        visual,head,train,ev=base.build_model(mode)
        initial=base.weight_state(visual,head,manifest['classes'],0,mode)
        before_train,before_eval=base.train_epoch,base.evaluate
        install(base,method)
        after_train,after_eval=before_train,before_eval
        eager_forward=base.forward
        compiled_forward=torch.compile(eager_forward,mode='reduce-overhead',dynamic=False)
        args=SimpleNamespace(data_root=root/'data/cls/plantvillage/color',batch_size=8,workers=4,seed=42,
                             mode=mode,epochs=150,smoke_steps=a.steps,label_smoothing=.05,
                             amp=torch.cuda.is_bf16_supported())
        loader=base.make_loader(args,manifest['splits']['train'],train,True)
        val=base.make_loader(args,manifest['splits']['val'][:256],ev,False)
        counts=base.np.bincount([r['label'] for r in manifest['splits']['train']],minlength=10)
        weights=torch.tensor(1/base.np.sqrt(counts),dtype=torch.float32,device='cuda');weights/=weights.mean()
        rng=base.get_rng();timings={'baseline':[],'optimized':[]};references=None
        # Alternate order to reduce temperature/clock drift. First pair also warms kernels.
        for trial in range(2):
            for name in (('baseline','optimized') if trial%2==0 else ('optimized','baseline')):
                base.load_weights(visual,head,initial,manifest['classes'],mode)
                params=[p for p in visual.parameters() if p.requires_grad]
                groups=([{'params':params,'lr':3e-4}] if params else [])+[{'params':head.parameters(),'lr':.003}]
                opt=torch.optim.AdamW(groups,weight_decay=.01)
                sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:base.lr_factor(step,150*len(loader),len(loader)))
                base.set_rng(rng)
                base.forward=eager_forward if name=='baseline' else compiled_forward
                torch.cuda.synchronize();start=time.perf_counter()
                stats=(before_train if name=='baseline' else after_train)(visual,head,loader,opt,sch,weights,args,1)
                metrics,logits,targets=(before_eval if name=='baseline' else after_eval)(visual,head,val,args.amp)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                snapshot={n:p.detach().cpu().clone() for n,p in visual.named_parameters() if p.requires_grad}
                snapshot.update({'head.'+n:p.detach().cpu().clone() for n,p in head.named_parameters()})
                if references is None:references=(snapshot,stats,logits,targets)
                else:
                    ref,rs,rl,ry=references
                    parameter_exact=all(torch.equal(snapshot[n],ref[n]) for n in snapshot)
                    max_parameter_difference=max((snapshot[n]-ref[n]).abs().max().item() for n in snapshot)
                    assert torch.equal(targets,ry),'Target mismatch'
                    print(json.dumps(dict(comparison=method,parameter_exact=parameter_exact,max_parameter_difference=max_parameter_difference,max_logit_difference=(logits-rl).abs().max().item(),loss_difference=abs(stats['loss']-rs['loss']))),flush=True)
                    
                timings[name].append(seconds)
                print(json.dumps(dict(method=method,trial=trial,variant=name,seconds=seconds,exact_match=(name=='baseline'))),flush=True)
        b=statistics.median(timings['baseline'][1:]);o=statistics.median(timings['optimized'][1:])
        results.append(dict(method=method,steps=a.steps,batch=8,precision='bf16' if args.amp else 'fp32',
                            baseline_seconds=b,optimized_seconds=o,speedup=b/o,timings=timings,
                            backend='torch.compile reduce-overhead',note='See log for numerical comparisons; not automatically approved for legacy resume'))
        a.output.write_text(json.dumps(results,indent=2))
        del visual,head,loader,val,opt,sch,groups,params,snapshot,references,initial
        torch.cuda.empty_cache()
    print('REPORT',a.output,flush=True)

if __name__=='__main__':main()

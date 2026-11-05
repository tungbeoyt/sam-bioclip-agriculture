"""Reduce metric/evaluation transfers; leave training math, data order and RNG intact."""
import hashlib
import json
from pathlib import Path

def install(base, method):
    torch,F=base.torch,base.F
    def train_epoch(visual, head, loader, optimizer, scheduler, weights, args, epoch):
        visual.train(True if method in ('resnet50','mobilenetv2') else args.mode=='lora')
        head.train()
        loader.generator.manual_seed(args.seed+epoch)
        parameters=[p for p in visual.parameters() if p.requires_grad]+list(head.parameters())
        # FP64 accumulation matches Python accumulation of FP32 scalar values.
        stats=torch.zeros(4,device='cuda',dtype=torch.float64)
        total=steps=0
        bar=base.tqdm(loader,desc=f'Train {epoch}/{args.epochs}',unit='batch',mininterval=.5)
        for step,(x,y) in enumerate(bar):
            if args.smoke_steps and step>=args.smoke_steps:break
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=base.forward(visual,head,x,args.amp)
            loss=F.cross_entropy(logits,y,weight=weights,label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; previous epoch checkpoint is preserved')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if step==0 and method in ('lora','vlm-vision','vlm-both'):
                if not any(p.grad is not None and torch.count_nonzero(p.grad).item()
                           for n,p in visual.named_parameters() if 'lora_B' in n):
                    raise RuntimeError('LoRA is not receiving nonzero gradients')
            optimizer.step();scheduler.step()
            n=len(y);total+=n;steps+=1
            stats[0]+=(logits.argmax(1)==y).sum()
            stats[1]+=loss.detach().double()*n
            stats[2]+=norm.detach().double()
            stats[3]=torch.maximum(stats[3],norm.detach().double())
            if step%10==0:
                correct,loss_sum,_,_=stats.tolist()
                bar.set_postfix(loss=f'{loss_sum/total:.4f}',acc=f'{correct/total:.1%}',
                                lr=f'{optimizer.param_groups[0]["lr"]:.2g}')
        correct,loss_sum,grad_sum,grad_max=stats.tolist()
        return dict(loss=loss_sum/total,accuracy=correct/total,
                    grad_norm_mean=grad_sum/steps,grad_norm_max=grad_max)

    @torch.no_grad()
    def evaluate(visual,head,loader,amp,desc='Validation',limit=0):
        visual.eval();head.eval()
        logits=[];targets=[]
        for step,(x,y) in enumerate(base.tqdm(loader,desc=desc,unit='batch',mininterval=1)):
            if limit and step>=limit:break
            logits.append(base.forward(visual,head,x.cuda(non_blocking=True),amp).float())
            targets.append(y)
        logits=torch.cat(logits).cpu();targets=torch.cat(targets)
        if not torch.isfinite(logits).all():raise RuntimeError('Non-finite evaluation logits')
        return dict(loss=F.cross_entropy(logits,targets).item(),
                    accuracy=(logits.argmax(1)==targets).float().mean().item()),logits,targets

    base.train_epoch=train_epoch
    base.evaluate=evaluate
    original_parse=base.parse_args
    def parse():
        args,state=original_parse()
        out=args.output_dir
        # Separate provenance, preserving legacy suite hashes and checkpoint format.
        out.mkdir(parents=True,exist_ok=True)
        path=out/'performance_sessions.json'
        records=json.loads(path.read_text()) if path.exists() else []
        records.append(dict(optimization='deferred metric transfers and batched evaluation transfer',
            runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            resumed=state is not None,resume_epoch=state.get('epoch') if state else None,
            batch=args.batch_size,precision=args.precision))
        base.save_json(path,records)
        return args,state
    base.parse_args=parse


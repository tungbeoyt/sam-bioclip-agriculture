"""GPU regression check for BF16 logits and FP32 weighted cross entropy."""
from pathlib import Path
import torch
import torch.nn.functional as F
from train_revision import load_base, configure_model


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this check')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required for this regression check')
    print('GPU:',torch.cuda.get_device_name(torch.cuda.current_device()))
    for method in ['vlm-vision','vlm-both']:
        base=load_base(Path(__file__).with_name('train_tomato_full_base.py'))
        configure_model(base,method,8)
        visual=torch.nn.Linear(8,10).cuda()
        x=torch.randn(8,8,device='cuda');y=torch.arange(8,device='cuda')
        weights=torch.ones(10,device='cuda',dtype=torch.float32)
        logits=base.forward(visual,torch.nn.Identity().cuda(),x,True)
        assert logits.dtype==torch.float32 and logits.device.type=='cuda'
        loss=F.cross_entropy(logits,y,weight=weights,label_smoothing=.05)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(visual.weight.grad).all()
        assert visual.weight.grad.abs().sum()>0
        print(method,'PASS',str(logits.device),str(logits.dtype),'loss',loss.item())


if __name__=='__main__':main()

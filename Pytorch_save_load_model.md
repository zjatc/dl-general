### Save model architecture and weights
```python
def save_checkpoint(ckpt_path, model, optimizer, epoch=0):
    ckpt = {
        'model': model.__class__(**kw),
        'weights': model.state_dict(),
        'opt_states': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(ckpt, ckpt_path)
```

### Load model architecture and weights
```python
def load_checkpoint(ckpt_path, is_eval=True, device=None):
    if device == 'cpu':
        device = torch.device('cpu')
    elif device == 'gpu':
        devcie = torch.device('cuda')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    # load architecture
    model = ckpt['model']
    # load trained weights
    model.load_state_dict(ckpt['weights'])
    if is_eval:
        for param in model.parameters():
            param.requires_grad = False
        opt = None
        current_epoch = 0
        model.eval()
    else:
        opt = OptimizerClass()
        opt.load_state_dict(ckpt['opt_states'])
        current_epoch = ckpt['epoch']
        model.train()
    if device.type == 'cuda':
        model.to(device)
    return model, device, opt, current_epoch
```

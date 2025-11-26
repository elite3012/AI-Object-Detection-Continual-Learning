import torch
from torch import nn, optim
from eval.metrics import accuracy
from eval.logger import log

def train_one_task(model, train_loader, test_loader, device="cuda", epochs=5, lr=0.1, 
                   replay_buffer=None, ewc=None, ewc_lambda=50000.0):
    """Train `model` on one task using a simple SGD loop.
    
    Args:
        replay_buffer: ReplayBuffer for experience replay (ER method)
        ewc: EWC regularizer (EWC method)
        ewc_lambda: EWC penalty weight
    
    Returns the trained model.
    """
    model.to(device)
    
    # Convert to float32 for stable training (prevents NaN with quantized models)
    original_dtype = next(model.parameters()).dtype
    if original_dtype == torch.float16:
        model = model.float()
    
    # Use AdamW for better convergence with replay buffers
    # LR tuned for continual learning with replay
    opt = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.999))
    # Gentle cosine decay (not aggressive step decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.00005)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Experience Replay: mix current batch with replay samples
            if replay_buffer is not None:
                replay_buffer.add_batch(x, y)
                # Sample 50% of batch size for balanced replay (proven to work)
                rx, ry = replay_buffer.sample(len(x) // 2)
                if rx is not None:
                    x = torch.cat([x, rx.to(device)])
                    y = torch.cat([y, ry.to(device)])
            
            logits = model(x)
            loss = ce(logits, y)
            
            # EWC: add regularization penalty
            if ewc is not None:
                ewc_penalty = ewc.penalty(model, lam=ewc_lambda)
                loss = loss + ewc_penalty
                if n_batches == 0:  # Log penalty on first batch
                    print(f"[EWC] Penalty: {ewc_penalty.item():.4f}")
            
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        acc = accuracy(model, test_loader, device)
        log(f"Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}")
        scheduler.step()  # Update learning rate
    
    # Keep model in float32 during continual learning
    # Only convert to float16 at the very end for deployment
    # if original_dtype == torch.float16:
    #     model = model.half()
    
    return model

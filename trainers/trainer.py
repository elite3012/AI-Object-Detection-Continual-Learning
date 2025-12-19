import torch
from torch import nn, optim
from eval.metrics import accuracy
from eval.logger import log

def train_one_task(model, train_loader, 
                   test_loader, 
                   device = "cuda", 
                   epochs = 5, 
                   lr = 0.1,
                   replay_buffer = None,
                   callback = None,
                   task_id = 0):
    """Train `model` on one task using a simple SGD loop.
    
    Args:
        replay_buffer: ReplayBuffer for experience replay (ER method)
        callback: Optional callback for UI updates (on_epoch_start, on_batch_end, on_epoch_end)
        task_id: Task ID for callback tracking
    
    Returns the trained model.
    """
    model.to(device)

    # Convert to float32 for stable training (prevents NaN with quantized models)
    original_dtype = next(model.parameters()).dtype
    if original_dtype == torch.float16:
        model = model.float()
    
    # Use AdamW for better convergence with replay buffers
    opt = optim.AdamW(params = model.parameters(),
                      lr = 0.0005,
                      weight_decay = 0.01,
                      betas = (0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = epochs, eta_min = 0.00005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        if callback:
            callback.on_epoch_start(task_id, epoch - 1)  # epoch is 1-indexed, convert to 0-indexed
        
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)

            # Experience Replay: mix current batch with replay samples
            if replay_buffer is not None:
                replay_buffer.add_batch(x_train, y_train)
                # Sample 70% of batch size for stronger retention
                replay_x_train, replay_y_train = replay_buffer.sample(int(len(x_train) * 0.7))
                if replay_x_train is not None:
                    x_train = torch.cat([x_train, replay_x_train.to(device)])
                    y_train = torch.cat([y_train, replay_y_train.to(device)])
            
            logits = model(x_train)
            loss = loss_fn(logits, y_train)

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            total_loss += loss.item()
            n_batches += 1
            
            if callback:
                callback.on_batch_end(task_id, epoch - 1, batch_idx, len(train_loader), loss.item())
    
        avg_loss = total_loss / max(n_batches, 1)
        
        # Evaluate only if test_loader is provided
        if test_loader is not None:
            acc = accuracy(model, test_loader, device)
            log(f"Epoch {epoch}: loss = {avg_loss:.4f}, acc = {acc:.4f}")
        else:
            log(f"Epoch {epoch}: loss = {avg_loss:.4f}")
        
        scheduler.step()
        
        if callback:
            callback.on_epoch_end(task_id, epoch - 1, avg_loss)
    
    return model

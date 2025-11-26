"""
Multi-Modal Continual Learning Trainer
Vision + Language training for Fashion-MNIST
"""
import torch
import torch.nn.functional as F
from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES
from data.fashion_text_descriptions import create_multimodal_batch, get_vocab_size
from eval.metrics import accuracy
from replay.buffer import ReplayBuffer


class MultiModalReplayBuffer:
    """Replay buffer that stores both images and text"""
    def __init__(self, m_per_class=500):
        self.m_per_class = m_per_class
        self.data = {}  # {class_id: [(image, text_tokens, label), ...]}
    
    def add_batch(self, images, text_tokens, labels):
        """Add multi-modal samples to buffer"""
        for img, text, label in zip(images, text_tokens, labels):
            cid = int(label)
            bucket = self.data.setdefault(cid, [])
            bucket.append((img.cpu(), text.cpu(), label.cpu()))
            if len(bucket) > self.m_per_class:
                bucket.pop(0)
    
    def sample(self, n):
        """Sample multi-modal batch"""
        all_samples = [s for samples in self.data.values() for s in samples]
        if not all_samples:
            return None, None, None
        
        import random
        batch = random.sample(all_samples, min(n, len(all_samples)))
        images, texts, labels = zip(*batch)
        
        return torch.stack(images), torch.stack(texts), torch.tensor(labels)


class MultiModalContinualTrainer:
    """
    Continual Learning with Multi-Modal (Vision + Language) Models
    Optimized for speed with mixed precision and efficient data loading
    """
    def __init__(self, model, use_replay=True, device="cuda", num_tasks=5, 
                 buffer_size=500, use_contrastive=False, use_amp=True):
        """
        Multi-modal continual learning trainer
        
        Args:
            model: Multi-modal model (CLIP-style or Fusion)
            use_replay: Use experience replay
            device: cuda or cpu
            num_tasks: Number of tasks
            buffer_size: Replay buffer size per class
            use_contrastive: Use contrastive loss (for CLIP models)
            use_amp: Use automatic mixed precision for speed (default: True)
        """
        self.model = model.to(device)
        self.device = device
        self.num_tasks = num_tasks
        self.use_replay = use_replay
        self.use_contrastive = use_contrastive
        self.use_amp = use_amp and device == "cuda"  # Only use AMP on GPU
        
        # Multi-modal replay buffer
        self.replay_buffer = MultiModalReplayBuffer(m_per_class=buffer_size) if use_replay else None
        
        # Track results
        self.task_acc_history = []
        self.all_test_loaders = []
        
        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.001, data_root="./data"):
        """Train on all tasks with multi-modal learning"""
        method_name = "Multi-Modal + ER" if self.use_replay else "Multi-Modal Finetune"
        print(f"\n{'='*60}")
        print(f"Multi-Modal Continual Learning: {method_name}")
        print(f"Training with Vision + Language")
        if self.use_replay:
            print(f"Replay Buffer: {self.replay_buffer.m_per_class} samples/class")
        print(f"{'='*60}\n")
        
        for task_id in range(self.num_tasks):
            print(f"\n{'='*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'='*60}")
            
            # Get data for current task
            train_loader, test_loader, classes = get_task_loaders_true_continual(
                task_id, batch_size=batch_size, root=data_root
            )
            self.all_test_loaders.append(test_loader)
            
            # Train on current task
            self._train_one_task(train_loader, epochs_per_task, lr)
            
            # Evaluate on all tasks
            task_accuracies = [0.0] * self.num_tasks
            for i in range(task_id + 1):
                acc = self._evaluate(self.all_test_loaders[i])
                task_accuracies[i] = acc
            
            print(f"[INFO] Task {task_id} complete: {task_accuracies[task_id]*100:.2f}%")
            if task_id > 0:
                prev_accs = ', '.join([f'{acc*100:.1f}%' for acc in task_accuracies[:task_id]])
                print(f"[INFO] Previous tasks: [{prev_accs}]")
            
            self.task_acc_history.append(task_accuracies)
        
        # Final results
        self._print_final_results()
        return self.task_acc_history
    
    def _train_one_task(self, train_loader, epochs=10, lr=0.001):
        """Train on one task with multi-modal data (optimized for speed)"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_cls_loss = 0.0
            total_contrast_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # CRITICAL FIX: Use RANDOM text descriptions during training!
                # This prevents label leakage while still training text encoder
                batch_size = images.size(0)
                random_labels = torch.randint(0, 10, (batch_size,))  # Random classes
                _, text_tokens, _ = create_multimodal_batch(images, random_labels)
                text_tokens = text_tokens.to(self.device)
                
                # Debug: Show text sample in first epoch
                if epoch == 0 and total == 0:
                    from data.fashion_text_descriptions import decode_tokens
                    print(f"[DEBUG] TRUE Label: {labels[0].item()}, RANDOM Text: {decode_tokens(text_tokens[0])}")
                    print(f"[DEBUG] This prevents label leakage - model must learn from vision!")
                
                
                # Experience replay (reduced to 50% for speed)
                if self.use_replay and self.replay_buffer:
                    replay_size = int(len(images) * 0.5)  # Reduced from 0.7 to 0.5
                    r_img, r_text, r_labels = self.replay_buffer.sample(replay_size)
                    
                    if r_img is not None:
                        images = torch.cat([images, r_img.to(self.device)])
                        text_tokens = torch.cat([text_tokens, r_text.to(self.device)])
                        labels = torch.cat([labels, r_labels.to(self.device)])
                    
                    # Add to buffer (first epoch only)
                    if epoch == 0:
                        orig_size = len(images) if r_img is None else len(images) - len(r_img)
                        self.replay_buffer.add_batch(
                            images[:orig_size], 
                            text_tokens[:orig_size], 
                            labels[:orig_size]
                        )
                
                optimizer.zero_grad()
                
                # Forward pass with automatic mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.use_contrastive and hasattr(self.model, 'contrastive_loss'):
                            # CLIP-style training with contrastive loss
                            img_feat, txt_feat = self.model(images, text_tokens, return_embeddings=True)
                            contrast_loss = self.model.contrastive_loss(img_feat, txt_feat)
                            
                            # Classification loss
                            logits = self.model(images)
                            cls_loss = F.cross_entropy(logits, labels)
                            
                            # Combined loss
                            loss = cls_loss + 0.3 * contrast_loss  # Reduced weight from 0.5
                            total_contrast_loss += contrast_loss.item()
                        else:
                            # Fusion model or classification only
                            if hasattr(self.model, 'forward') and 'text_tokens' in self.model.forward.__code__.co_varnames:
                                logits = self.model(images, text_tokens)
                            else:
                                logits = self.model(images)
                            
                            cls_loss = F.cross_entropy(logits, labels)
                            loss = cls_loss
                    
                    # Backward with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard training without AMP
                    if self.use_contrastive and hasattr(self.model, 'contrastive_loss'):
                        img_feat, txt_feat = self.model(images, text_tokens, return_embeddings=True)
                        contrast_loss = self.model.contrastive_loss(img_feat, txt_feat)
                        logits = self.model(images)
                        cls_loss = F.cross_entropy(logits, labels)
                        loss = cls_loss + 0.3 * contrast_loss
                        total_contrast_loss += contrast_loss.item()
                    else:
                        if hasattr(self.model, 'forward') and 'text_tokens' in self.model.forward.__code__.co_varnames:
                            logits = self.model(images, text_tokens)
                        else:
                            logits = self.model(images)
                        cls_loss = F.cross_entropy(logits, labels)
                        loss = cls_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)
            
            scheduler.step()
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_loader)
                avg_cls_loss = total_cls_loss / len(train_loader)
                train_acc = 100. * correct / total
                
                if self.use_contrastive:
                    avg_contrast = total_contrast_loss / len(train_loader)
                    print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} (cls: {avg_cls_loss:.4f}, contrast: {avg_contrast:.4f}) | Acc: {train_acc:.2f}%")
                else:
                    print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%")
    
    def _evaluate(self, test_loader):
        """Evaluate on test set - FIXED: No label leakage via text"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # CRITICAL FIX: Use vision-only inference for fair evaluation
                # Multi-modal models should classify from IMAGE, not from text hints
                if hasattr(self.model, 'encode_image'):
                    # CLIP-style: Use class text embeddings (no label leak)
                    logits = self.model(images, text_tokens=None)
                elif hasattr(self.model, 'forward') and 'text_tokens' in self.model.forward.__code__.co_varnames:
                    # Fusion model: Create generic/dummy text (no label info)
                    # Use a neutral description that doesn't reveal the class
                    batch_size = images.size(0)
                    # Generic text: "a piece of clothing or footwear"
                    dummy_tokens = torch.zeros(batch_size, 32, dtype=torch.long).to(self.device)
                    dummy_tokens[:, 0] = 2  # START token
                    dummy_tokens[:, 1] = 4  # 'a'
                    dummy_tokens[:, 2] = 18  # 'clothing'
                    dummy_tokens[:, 3] = 3  # END token
                    logits = self.model(images, dummy_tokens)
                else:
                    logits = self.model(images)
                
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)
        
        return correct / total if total > 0 else 0.0
    
    def _print_final_results(self):
        """Print final results"""
        print(f"\n{'='*60}")
        print(f"FINAL MULTI-MODAL RESULTS")
        print(f"{'='*60}")
        
        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)
        
        for i, acc in enumerate(final_accs):
            print(f"  Task {i} ({TASK_THEMES.get(i, f'Task {i}')}): {acc*100:.2f}%")
        
        print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")
        
        # Compute forgetting
        if self.num_tasks > 1:
            forgetting = 0.0
            for i in range(self.num_tasks - 1):
                max_acc = max([self.task_acc_history[j][i] for j in range(i, self.num_tasks)])
                forgetting += (max_acc - final_accs[i])
            forgetting /= (self.num_tasks - 1)
            print(f"  Forgetting: {forgetting*100:.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def get_metrics(self):
        """Get evaluation metrics"""
        if not self.task_acc_history:
            return {}
        
        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)
        
        forgetting = 0.0
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                max_acc = max([self.task_acc_history[j][i] for j in range(i, self.num_tasks)])
                forgetting += (max_acc - final_accs[i])
            forgetting /= (self.num_tasks - 1)
        
        return {
            "average_accuracy": avg_acc,
            "forgetting": forgetting,
            "final_accuracies": final_accs,
            "accuracy_matrix": self.task_acc_history
        }

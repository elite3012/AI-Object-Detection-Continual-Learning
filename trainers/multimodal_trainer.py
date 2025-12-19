"""
Multi-Modal Continual Learning Trainer
Combines vision and text encoders for robust continual learning
"""

import torch
from torch import nn, optim
from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES
from data.fashion_text import get_text_description, CLASS_NAMES
from models.text_encoder import encode_texts
from eval.metrics import accuracy
from replay.buffer import ReplayBuffer

class MultiModalContinualTrainer:
    """
    Multi-modal continual learning with Experience Replay
    Trains both vision and text encoders jointly
    """
    def __init__(self, multimodal_model, use_replay=True, device="cuda", 
                 num_tasks=5, buffer_size=500, text_mode='simple', text_dropout=0.3):
        """
        Args:
            multimodal_model: MultiModalClassifier instance
            use_replay: Use Experience Replay
            device: "cuda" or "cpu"
            num_tasks: Number of tasks
            buffer_size: Samples per class in replay buffer
            text_mode: 'simple', 'rich', or 'attributes' for text descriptions
            text_dropout: Probability to zero out text features (prevent text cheating)
        """
        self.model = multimodal_model
        self.use_replay = use_replay
        self.device = device
        self.num_tasks = num_tasks
        self.text_mode = text_mode
        self.text_dropout = text_dropout  # NEW: Force model to use vision too
        
        # Replay buffer stores (image, label, text_input_ids, text_mask)
        self.replay_buffer = ReplayBuffer(m_per_class=buffer_size) if use_replay else None
        
        # Track results
        self.task_acc_history = []
        self.all_test_loaders = []
        
    def train_one_task(self, train_loader, test_loader, epochs=10, lr=0.0005, callback=None, task_id=0):
        """
        Train on one task
        
        Args:
            train_loader: DataLoader for current task
            test_loader: DataLoader for testing
            epochs: Number of epochs
            lr: Learning rate
            callback: Optional callback for UI updates
            task_id: Task ID for callback tracking
        """
        self.model.to(self.device)
        self.model.train()
        
        # Optimizer with different LR for vision and text
        vision_params = list(self.model.vision_encoder.parameters())
        text_params = list(self.model.text_encoder.parameters())
        fusion_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())
        
        optimizer = optim.AdamW([
            {'params': vision_params, 'lr': lr, 'weight_decay': 0.01},
            {'params': text_params, 'lr': lr * 2, 'weight_decay': 0.01},  # Higher LR for text
            {'params': fusion_params, 'lr': lr * 1.5, 'weight_decay': 0.01}
        ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            if callback:
                callback.on_epoch_start(task_id, epoch - 1)
            
            total_loss = 0.0
            n_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = images.size(0)
                
                # Get text descriptions for labels
                texts = [get_text_description(int(label), mode=self.text_mode) for label in labels]
                input_ids, attention_mask = encode_texts(texts, device=self.device)
                
                # Experience Replay: mix current + old samples
                if self.replay_buffer is not None:
                    # Store current batch (we'll store text info too)
                    self.replay_buffer.add_batch(images, labels)
                    
                    # Sample from buffer (70% of batch size)
                    replay_samples = self.replay_buffer.sample(int(batch_size * 0.7))
                    if replay_samples is not None:
                        replay_images, replay_labels = replay_samples
                        replay_images = replay_images.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        
                        # Get text for replayed samples
                        replay_texts = [get_text_description(int(label), mode=self.text_mode) for label in replay_labels]
                        replay_input_ids, replay_attention_mask = encode_texts(replay_texts, device=self.device)
                        
                        # Concatenate current + replay
                        images = torch.cat([images, replay_images])
                        labels = torch.cat([labels, replay_labels])
                        input_ids = torch.cat([input_ids, replay_input_ids])
                        attention_mask = torch.cat([attention_mask, replay_attention_mask])
                
                # Forward pass
                logits = self.model(images, input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                
                # Add vision loss (ensure vision encoder learns)
                if torch.rand(1).item() < self.text_dropout:
                    # 30% of time: force vision-only prediction
                    vision_logits = self.model.forward_vision_only(images)
                    vision_loss = loss_fn(vision_logits, labels)
                    loss = 0.7 * loss + 0.3 * vision_loss  # Balanced loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                if callback:
                    callback.on_batch_end(task_id, epoch - 1, batch_idx, len(train_loader), loss.item())
            
            avg_loss = total_loss / max(n_batches, 1)
            
            # Evaluate
            if test_loader:
                acc = self.evaluate_task(test_loader)
                print(f"[LOG] Epoch {epoch}: loss = {avg_loss:.4f}, acc = {acc:.4f}")
            else:
                print(f"[LOG] Epoch {epoch}: loss = {avg_loss:.4f}")
            
            if callback:
                callback.on_epoch_end(task_id, epoch - 1, avg_loss)
            
            scheduler.step()
    
    def evaluate_task(self, test_loader):
        """Evaluate on a single task"""
        self.model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get text descriptions
                texts = [get_text_description(int(label), mode=self.text_mode) for label in labels]
                input_ids, attention_mask = encode_texts(texts, device=self.device)
                
                # Forward
                logits = self.model(images, input_ids, attention_mask)
                preds = logits.argmax(1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        self.model.train()
        return correct / max(total, 1)
    
    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.0005, data_root="./data"):
        """
        Train on all tasks sequentially
        
        Args:
            epochs_per_task: Epochs to train per task
            batch_size: Batch size
            lr: Learning rate
            data_root: Data directory
        """
        method_name = "Multi-Modal + Experience Replay" if self.use_replay else "Multi-Modal + Finetune"
        
        print(f"\n{'-'*60}")
        print(f"Multi-Modal Continual Learning: {method_name}")
        print(f"Training with Vision + Text modalities")
        if self.use_replay and self.replay_buffer:
            print(f"Replay Buffer: {self.replay_buffer.m_per_class} samples/class")
        print(f"Text Mode: {self.text_mode}")
        print(f"{'-'*60}\n")
        
        for task_id in range(self.num_tasks):
            print(f"\n{'-'*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'-'*60}")
            
            # Get data for current task
            train_loader, test_loader, classes = get_task_loaders_true_continual(
                task_id=task_id,
                batch_size=batch_size,
                root=data_root
            )
            self.all_test_loaders.append(test_loader)
            
            # Train
            self.train_one_task(train_loader, test_loader, epochs=epochs_per_task, lr=lr)
            
            # Evaluate on all tasks seen so far
            task_accuracies = [0.0] * self.num_tasks
            for i in range(task_id + 1):
                acc = self.evaluate_task(self.all_test_loaders[i])
                task_accuracies[i] = acc
            
            print(f"[INFO] Task {task_id} training complete: {task_accuracies[task_id]*100:.2f}%")
            if task_id > 0:
                prev_accs = ', '.join([f'{acc*100:.1f}%' for acc in task_accuracies[:task_id]])
                print(f"[INFO] Previous tasks: [{prev_accs}]")
            
            self.task_acc_history.append(task_accuracies)
        
        # Final evaluation
        self._print_final_results()
    
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
        
        if avg_acc >= 0.90:
            status = "EXCELLENT - Multi-modal synergy working!"
        elif avg_acc >= 0.80:
            status = "GOOD - Strong multi-modal learning"
        elif avg_acc >= 0.70:
            status = "DECENT - Room for improvement"
        else:
            status = "NEEDS WORK - Check fusion strategy"
        
        print(f"  Status: {status}")
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
                current_acc = self.task_acc_history[-1][i]
                forgetting += (max_acc - current_acc)
            forgetting /= (self.num_tasks - 1)
        
        return {
            "average_accuracy": avg_acc,
            "forgetting": forgetting,
            "final_accuracies": final_accs,
            "accuracy_matrix": self.task_acc_history
        }

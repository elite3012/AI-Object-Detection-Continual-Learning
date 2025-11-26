"""
PEFT Continual Learning Trainer
Combines TRUE Continual Learning with Parameter-Efficient Fine-Tuning (LoRA)
"""
import torch
from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES
from trainers.trainer import train_one_task
from eval.metrics import accuracy
from replay.buffer import ReplayBuffer
from models.peft_lora import apply_lora_to_model, get_lora_parameters


class PEFTContinualTrainer:
    """
    TRUE Continual Learning with PEFT (LoRA)
    Only trains LoRA adapters (1-10% of parameters)
    """
    def __init__(self, model, use_replay=True, device="cuda", num_tasks=5, 
                 buffer_size=500, lora_rank=24, lora_alpha=48):
        """
        PEFT Continual Learning trainer
        
        Args:
            model: Foundation model (ViT, ResNet, etc.)
            use_replay: Use Experience Replay
            device: "cuda" or "cpu"
            num_tasks: Number of tasks
            buffer_size: Samples per class in replay buffer (default: 500)
            lora_rank: LoRA rank (16-32, higher=better capacity for hard tasks)
            lora_alpha: LoRA scaling factor (32-64, 2x rank is optimal)
        """
        self.device = device
        self.num_tasks = num_tasks
        self.use_replay = use_replay
        
        # Apply LoRA to model
        print(f"\n{'='*60}")
        print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
        print(f"{'='*60}")
        
        self.model, trainable_params, total_params = apply_lora_to_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=None,  # Apply to all Linear layers
            dropout=0.0  # No dropout for maximum capacity
        )
        
        self.trainable_params = trainable_params
        self.total_params = total_params
        
        # Experience Replay buffer with larger capacity
        self.replay_buffer = ReplayBuffer(m_per_class=buffer_size) if use_replay else None
        
        # Track results
        self.task_acc_history = []
        self.all_test_loaders = []
        
    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.001, data_root="./data"):
        """
        Train on all tasks with PEFT
        
        Note: lr is typically lower for LoRA (0.001 vs 0.01 for full fine-tuning)
        """
        method_name = "LoRA + Experience Replay" if self.use_replay else "LoRA + Finetune"
        print(f"\n{'='*60}")
        print(f"PEFT Continual Learning: {method_name}")
        print(f"Training ONLY on new classes per task")
        if self.use_replay and self.replay_buffer:
            print(f"Replay Buffer: {self.replay_buffer.m_per_class} samples/class")
        print(f"Trainable: {self.trainable_params:,} / {self.total_params:,} params")
        print(f"{'='*60}\n")
        
        for task_id in range(self.num_tasks):
            print(f"\n{'='*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'='*60}")
            
            # Get data for CURRENT task only
            train_loader, test_loader, classes = get_task_loaders_true_continual(
                task_id, batch_size=batch_size, root=data_root
            )
            self.all_test_loaders.append(test_loader)
            
            # Train with PEFT (only LoRA parameters are trained)
            self.model = self._train_one_task_peft(
                train_loader,
                test_loader,
                epochs=epochs_per_task,
                lr=lr
            )
            
            # Evaluate on ALL tasks seen so far
            task_accuracies = [0.0] * self.num_tasks
            for i in range(task_id + 1):
                acc = accuracy(self.model, self.all_test_loaders[i], device=self.device)
                task_accuracies[i] = acc
            
            print(f"[INFO] Task {task_id} training complete: {task_accuracies[task_id]*100:.2f}%")
            if task_id > 0:
                prev_accs = ', '.join([f'{acc*100:.1f}%' for acc in task_accuracies[:task_id]])
                print(f"[INFO] Previous tasks: [{prev_accs}]")
            
            self.task_acc_history.append(task_accuracies)
        
        # Final evaluation
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        final_accuracies = self.task_acc_history[-1]
        for i in range(len(self.all_test_loaders)):
            print(f"  Task {i} ({TASK_THEMES.get(i, f'Task {i}')}): {final_accuracies[i]*100:.2f}%")
        
        self._print_results(final_accuracies)
        
        return self.task_acc_history
    
    def _train_one_task_peft(self, train_loader, test_loader, epochs=10, lr=0.001):
        """Train one task with PEFT optimizer"""
        # Only optimize LoRA parameters
        lora_params = get_lora_parameters(self.model)
        
        if len(lora_params) == 0:
            # Fallback: train all trainable params
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use higher initial LR with warmup for better convergence
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
        
        # Warmup + Cosine schedule for stable training
        def lr_lambda(epoch):
            warmup_epochs = 2
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Experience Replay: Mix with MORE old samples (70% replay ratio)
                # IMPORTANT: Sample BEFORE adding new data
                if self.use_replay and self.replay_buffer:
                    # Sample 70% of batch size from old tasks (more aggressive replay)
                    replay_size = int(len(data) * 0.7)
                    rx, ry = self.replay_buffer.sample(replay_size)
                    if rx is not None:
                        # Mix: current batch + replay samples
                        data = torch.cat([data, rx.to(self.device)])
                        target = torch.cat([target, ry.to(self.device)])
                    
                    # Add current batch to buffer AFTER sampling (only on first epoch)
                    if epoch == 0:
                        original_size = len(data) if rx is None else len(data) - len(rx)
                        self.replay_buffer.add_batch(data[:original_size], target[:original_size])
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
            
            scheduler.step()
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_loader)
                train_acc = 100. * correct / total
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {current_lr:.6f}")
        
        return self.model
    
    def _print_results(self, final_accuracies):
        """Print final results"""
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS (PEFT)")
        print(f"{'='*60}")
        
        avg_acc = sum(final_accuracies) / len(final_accuracies)
        print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")
        print(f"  Trainable Params: {self.trainable_params:,} ({100*self.trainable_params/self.total_params:.2f}%)")
        print(f"  Memory Efficiency: {self.total_params/self.trainable_params:.1f}x reduction")
        
        if avg_acc >= 0.90:
            print(f"  Status: EXCELLENT - Achieved >90% with PEFT!")
        elif avg_acc >= 0.80:
            print(f"  Status: GOOD - High accuracy with efficient training")
        elif avg_acc >= 0.70:
            print(f"  Status: DECENT - Consider increasing LoRA rank")
        else:
            print(f"  Status: NEEDS WORK - Try higher rank or more epochs")
        
        print(f"\n{'='*60}\n")
    
    def get_metrics(self):
        """Compute metrics"""
        if not self.task_acc_history:
            return {}
        
        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)
        
        # Backward Transfer
        bwt = 0.0
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                bwt += (self.task_acc_history[-1][i] - self.task_acc_history[i][i])
            bwt /= (self.num_tasks - 1)
        
        # Forgetting
        forgetting = 0.0
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                max_acc = max([self.task_acc_history[j][i] for j in range(i, self.num_tasks)])
                current_acc = self.task_acc_history[-1][i]
                forgetting += (max_acc - current_acc)
            forgetting /= (self.num_tasks - 1)
        
        return {
            "average_accuracy": avg_acc,
            "backward_transfer": bwt,
            "forgetting": forgetting,
            "final_accuracies": final_accs,
            "accuracy_matrix": self.task_acc_history,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "efficiency_ratio": self.total_params / self.trainable_params
        }

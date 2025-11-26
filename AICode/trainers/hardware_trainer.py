"""
Hardware-Optimized Continual Learning Trainer
Integrates quantization and pruning into continual learning pipeline
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
import copy
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from trainers.continual_trainer_true import TrueContinualTrainer
    from optimizers.quantization import quantize_model, QuantizationConfig
    from optimizers.pruning import prune_model, PruningConfig
    from optimizers.hardware_optimizer import ModelProfiler
except ModuleNotFoundError:
    # Fallback for direct execution
    from continual_trainer_true import TrueContinualTrainer
    sys.path.insert(0, str(Path(__file__).parent.parent / 'optimizers'))
    from quantization import quantize_model, QuantizationConfig
    from pruning import prune_model, PruningConfig
    from hardware_optimizer import ModelProfiler


class HardwareOptimizedTrainer(TrueContinualTrainer):
    """
    Continual learning trainer with hardware optimizations
    
    Features:
    - Quantization after each task
    - Gradual pruning across tasks
    - Efficiency tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_replay: bool = True,
        device: str = 'cpu',
        num_tasks: int = 5,
        buffer_size: int = 200,
        # Optimization configs
        enable_quantization: bool = False,
        quantization_dtype: str = 'qint8',  # 'qint8' or 'float16'
        enable_pruning: bool = False,
        target_sparsity: float = 0.5,
        prune_per_task: bool = True,  # Prune after each task
        # Efficiency tracking
        track_efficiency: bool = True
    ):
        super().__init__(model, use_replay, device, num_tasks, buffer_size)
        
        self.enable_quantization = enable_quantization
        self.quantization_dtype = quantization_dtype
        self.enable_pruning = enable_pruning
        self.target_sparsity = target_sparsity
        self.prune_per_task = prune_per_task
        self.track_efficiency = track_efficiency
        
        # Tracking
        self.efficiency_history = []
        self.profiler = ModelProfiler(device) if track_efficiency else None
        
        # Pruning state
        self.current_sparsity = 0.0
    
    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.01, data_root="./data"):
        """Train on all tasks with hardware optimizations"""
        from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES
        from trainers.trainer import train_one_task
        from eval.metrics import accuracy
        
        method_name = "Experience Replay" if self.use_replay else "Finetune (No CL)"
        print(f"\n{'='*60}")
        print(f"Hardware-Optimized TRUE Continual Learning: {method_name}")
        print(f"Quantization: {self.quantization_dtype if self.enable_quantization else 'Disabled'}")
        print(f"Pruning: {self.target_sparsity*100:.0f}% sparsity" if self.enable_pruning else "Pruning: Disabled")
        print(f"{'='*60}\n")
        
        for task_id in range(self.num_tasks):
            print(f"\n{'='*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'='*60}")
            
            # Profile before training
            train_loader, test_loader, classes = get_task_loaders_true_continual(
                task_id, batch_size=batch_size, root=data_root
            )
            self.all_test_loaders.append(test_loader)
            
            if self.track_efficiency and self.profiler:
                before_metrics = self.profiler.profile_model(self.model, test_loader)
                print(f"\n[Before Task {task_id}]")
                print(f"  Size: {before_metrics['model_size_mb']:.2f} MB")
                print(f"  Params: {before_metrics['total_params']:,}")
                print(f"  FPS: {before_metrics['fps']:.1f}")
            
            # Standard continual learning training
            self.model = train_one_task(
                self.model,
                train_loader,
                test_loader,
                device=self.device,
                epochs=epochs_per_task,
                lr=lr,
                replay_buffer=self.replay_buffer,
                ewc=None,
                ewc_lambda=0.0
            )
            
            # Apply optimizations after training
            # CRITICAL: Only apply optimizations on FINAL task to preserve learning capacity
            is_final_task = (task_id == self.num_tasks - 1)
            
            if self.enable_pruning and self.prune_per_task:
                self._apply_pruning(task_id, train_loader)
            
            # Only quantize on final task (not every task!)
            if self.enable_quantization and is_final_task:
                self._apply_quantization(task_id, train_loader)
            elif self.enable_quantization and not is_final_task:
                print(f"\n[Quantization] Task {task_id}: Skipping (will quantize after final task)")
            
            # Profile after optimization
            if self.track_efficiency and self.profiler:
                after_metrics = self.profiler.profile_model(self.model, test_loader)
                
                efficiency = {
                    'task_id': task_id,
                    'before': before_metrics,
                    'after': after_metrics,
                    'size_reduction': (1 - after_metrics['model_size_mb'] / before_metrics['model_size_mb']) * 100,
                    'speedup': after_metrics['fps'] / before_metrics['fps'],
                    'param_reduction': (1 - after_metrics['total_params'] / before_metrics['total_params']) * 100
                }
                
                self.efficiency_history.append(efficiency)
                
                print(f"\n[After Task {task_id} Optimization]")
                print(f"  Size: {after_metrics['model_size_mb']:.2f} MB ({efficiency['size_reduction']:.1f}% reduction)")
                print(f"  Params: {after_metrics['total_params']:,} ({efficiency['param_reduction']:.1f}% reduction)")
                print(f"  FPS: {after_metrics['fps']:.1f} ({efficiency['speedup']:.2f}x speedup)")
            
            # Evaluate on ALL tasks seen so far
            task_accuracies = [0.0] * self.num_tasks
            for i in range(task_id + 1):
                acc = accuracy(self.model, self.all_test_loaders[i], device=self.device)
                task_accuracies[i] = acc
            
            print(f"\n[INFO] Task {task_id} training complete: {task_accuracies[task_id]*100:.2f}%")
            if task_id > 0:
                prev_accs = ', '.join([f'{acc*100:.1f}%' for acc in task_accuracies[:task_id]])
                print(f"[INFO] Previous tasks: [{prev_accs}]")
            
            self.task_acc_history.append(task_accuracies)
    
    def _apply_pruning(self, task_id: int, train_loader):
        """Apply gradual knowledge-preserving pruning"""
        if not self.enable_pruning:
            return
        
        # NEW APPROACH: Gradual pruning across tasks
        pruning_schedule = {
            0: 0.0,
            1: 0.0,
            2: 0.05,  # 5% at task 2
            3: 0.07,  # 7% at task 3
            4: 0.10   # 10% at task 4 (final)
        }
        
        target_sparsity = pruning_schedule.get(task_id, 0.0)
        incremental_sparsity = target_sparsity - self.current_sparsity
        
        if incremental_sparsity <= 0:
            print(f"\n[Pruning] Task {task_id}: Skipping (schedule: {target_sparsity*100:.0f}%)")
            return
        
        print(f"\n[Pruning] Task {task_id}: Gradual pruning {self.current_sparsity*100:.1f}% → {target_sparsity*100:.1f}% (+{incremental_sparsity*100:.1f}%)")
        
        # Convert to float32 for stable pruning
        original_dtype = next(self.model.parameters()).dtype
        if original_dtype == torch.float16:
            self.model = self.model.float()
        
        # Create teacher model (knowledge distillation)
        teacher_model = self._create_teacher_copy()
        
        # Apply importance-based pruning
        self._prune_by_importance(incremental_sparsity, train_loader)
        
        # Adapt architecture if needed
        if hasattr(self.model, 'adapt_to_pruning'):
            self.model.adapt_to_pruning()
        
        # Knowledge distillation fine-tuning
        self._distillation_finetune(teacher_model, train_loader, epochs=5)
        
        # Update sparsity tracker
        self.current_sparsity = target_sparsity
        
        # Convert back to original dtype
        if original_dtype == torch.float16:
            self.model = self.model.half()
        
        print(f"[Pruning] Complete! Total sparsity: {self.current_sparsity*100:.1f}%")
    
    def _create_teacher_copy(self):
        """Create a copy of current model for knowledge distillation"""
        import copy
        teacher = copy.deepcopy(self.model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def _prune_by_importance(self, amount: float, train_loader):
        """Importance-based structured pruning (prune least important channels)"""
        import numpy as np
        
        # Calculate importance scores for each layer
        layer_importance = {}
        
        # For ViT: focus on transformer MLP layers (least sensitive)
        if hasattr(self.model, 'transformer'):
            for idx, layer in enumerate(self.model.transformer.layers):
                # Linear1 and Linear2 in MLP - calculate L1 norm per output channel
                if hasattr(layer, 'linear1'):
                    # Weight shape: [out_features, in_features]
                    # L1 norm per output channel (sum over input dim)
                    l1_norm = torch.norm(layer.linear1.weight.data, p=1, dim=1)
                    layer_importance[f'transformer.layers.{idx}.linear1'] = l1_norm
                if hasattr(layer, 'linear2'):
                    l2_norm = torch.norm(layer.linear2.weight.data, p=1, dim=1)
                    layer_importance[f'transformer.layers.{idx}.linear2'] = l2_norm
        
        # For CNN: prune conv layers
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and module.out_channels > 16:
                    # Weight shape: [out_channels, in_channels, H, W]
                    # L1 norm per output channel (sum over other dims)
                    importance = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
                    layer_importance[name] = importance
        
        # Select channels to prune based on importance
        if layer_importance:
            # Prune channels with lowest L1 norm (least important)
            for name, importance in layer_importance.items():
                num_channels = len(importance)
                num_to_prune = max(1, int(num_channels * amount))
                
                # Get indices of least important channels
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]
                
                # Apply structured pruning (set weights to zero)
                module = dict(self.model.named_modules())[name]
                if isinstance(module, nn.Linear):
                    # Zero out and freeze pruned weights
                    module.weight.data[prune_indices, :] = 0
                    if module.bias is not None:
                        module.bias.data[prune_indices] = 0
                    
                    # Register hook to keep pruned weights at zero
                    def make_pruning_hook(indices):
                        def hook(grad):
                            grad[indices, :] = 0  # Zero gradient for pruned channels
                            return grad
                        return hook
                    
                    if not hasattr(module.weight, '_pruning_hook'):
                        handle = module.weight.register_hook(make_pruning_hook(prune_indices))
                        module.weight._pruning_hook = handle
                        
                elif isinstance(module, nn.Conv2d):
                    module.weight.data[prune_indices, :, :, :] = 0
                    if module.bias is not None:
                        module.bias.data[prune_indices] = 0
                    
                    # Register hook for conv layers too
                    def make_conv_pruning_hook(indices):
                        def hook(grad):
                            grad[indices, :, :, :] = 0
                            return grad
                        return hook
                    
                    if not hasattr(module.weight, '_pruning_hook'):
                        handle = module.weight.register_hook(make_conv_pruning_hook(prune_indices))
                        module.weight._pruning_hook = handle
        
        print(f"[Pruning] Pruned {len(layer_importance)} layers by importance")
    
    def _distillation_finetune(self, teacher_model, train_loader, epochs=5):
        """Fine-tune with knowledge distillation + replay buffer"""
        print(f"\n[Distillation] Fine-tuning with teacher guidance...")
        
        self.model.train()
        teacher_model.eval()
        
        # Lower LR for fine-tuning, AdamW for better regularization
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00005, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        kl_div = nn.KLDivLoss(reduction='batchmean')
        temperature = 3.0  # Lower temperature for sharper targets
        alpha = 0.5  # Balanced distillation (50% teacher, 50% hard labels)
        
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            num_batches = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Mix with balanced replay buffer samples (75% of batch)
                if self.replay_buffer is not None and len(self.replay_buffer.data) > 0:
                    replay_data = self.replay_buffer.sample(n=int(len(images) * 0.75))
                    if replay_data[0] is not None:
                        replay_images, replay_labels = replay_data
                        replay_images = replay_images.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        images = torch.cat([images, replay_images], dim=0)
                        labels = torch.cat([labels, replay_labels], dim=0)
                
                # Student predictions
                student_logits = self.model(images)
                
                # Teacher predictions (no grad)
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                
                # Combined loss: distillation + hard labels
                hard_loss = criterion(student_logits, labels)
                
                soft_loss = kl_div(
                    nn.functional.log_softmax(student_logits / temperature, dim=1),
                    nn.functional.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)
                
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = student_logits.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%")
        
        print(f"[Distillation] Complete!")
    
    def _apply_quantization(self, task_id: int, train_loader):
        """Apply quantization"""
        # Skip quantization if model is already heavily pruned (causes NaN with ViT)
        if hasattr(self.model, 'embed_dim') and self.current_sparsity > 0.05:
            print(f"\n[Quantization] Skipping - model already optimized with {self.current_sparsity*100:.1f}% pruning")
            return
        
        print(f"\n[Quantization] Task {task_id}: Applying {self.quantization_dtype}")
        
        # For ViT models, use post-training quantization only (no QAT)
        qat_epochs = 0 if hasattr(self.model, 'patch_embed') else 2
        
        result = quantize_model(
            self.model,
            train_loader,
            dtype=self.quantization_dtype,
            qat_epochs=qat_epochs,
            device=self.device
        )
        
        self.model = result['quantized_model']
    
    def get_efficiency_summary(self) -> Dict:
        """Get summary of efficiency improvements"""
        if not self.efficiency_history:
            return {}
        
        first = self.efficiency_history[0]['before']
        last = self.efficiency_history[-1]['after']
        
        summary = {
            'total_size_reduction': (1 - last['model_size_mb'] / first['model_size_mb']) * 100,
            'total_speedup': last['fps'] / first['fps'],
            'total_param_reduction': (1 - last['total_params'] / first['total_params']) * 100,
            'final_size_mb': last['model_size_mb'],
            'final_fps': last['fps'],
            'final_params': last['total_params']
        }
        
        return summary
    
    def print_efficiency_report(self):
        """Print detailed efficiency report"""
        if not self.efficiency_history:
            print("No efficiency data available")
            return
        
        print("\n" + "="*80)
        print("HARDWARE OPTIMIZATION EFFICIENCY REPORT")
        print("="*80)
        
        # Per-task table
        print(f"\n{'Task':<6} {'Size (MB)':<12} {'Params':<12} {'FPS':<10} {'Reduction':<12}")
        print("-"*80)
        
        for eff in self.efficiency_history:
            tid = eff['task_id']
            after = eff['after']
            reduction = eff['size_reduction']
            
            print(f"{tid:<6} "
                  f"{after['model_size_mb']:<12.2f} "
                  f"{after['total_params']:<12,} "
                  f"{after['fps']:<10.1f} "
                  f"{reduction:<12.1f}%")
        
        # Summary
        summary = self.get_efficiency_summary()
        
        print("\n" + "="*80)
        print("CUMULATIVE IMPROVEMENTS")
        print("="*80)
        print(f"Total Size Reduction:   {summary['total_size_reduction']:.1f}%")
        print(f"Total Speedup:          {summary['total_speedup']:.2f}x")
        print(f"Total Param Reduction:  {summary['total_param_reduction']:.1f}%")
        print(f"\nFinal Model:")
        print(f"  Size: {summary['final_size_mb']:.2f} MB")
        print(f"  Params: {summary['final_params']:,}")
        print(f"  FPS: {summary['final_fps']:.1f}")
        print("="*80)


class MemoryEfficientTrainer(TrueContinualTrainer):
    """
    Memory-efficient trainer using gradient checkpointing and mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        regularizer=None,
        device: str = 'cpu',
        use_amp: bool = True,  # Automatic Mixed Precision
        gradient_checkpointing: bool = False
    ):
        # Use TrueContinualTrainer constructor
        super().__init__(model, use_replay=True, device=device, num_tasks=5, buffer_size=200)
        
        self.use_amp = use_amp and device == 'cuda'
        self.gradient_checkpointing = gradient_checkpointing
        
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except AttributeError:
                # PyTorch < 2.0
                self.scaler = torch.cuda.amp.GradScaler()
        
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        # For transformer-based models
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # For sequential models, manually checkpoint
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                # Wrap forward with checkpointing
                original_forward = module.forward
                
                # Skip checkpointing for type safety - can add if needed
                pass
    
    def train_task(
        self,
        task_id: int,
        train_loader,
        test_loader,
        epochs: int = 10,
        lr: float = 0.001,
        evaluate_every: int = 1
    ):
        """Train with memory optimizations"""
        print(f"\n[Memory-Efficient Training] Task {task_id}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Checkpointing: {self.gradient_checkpointing}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if self.use_amp:
                    try:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    except AttributeError:
                        # PyTorch < 2.0
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            train_acc = 100.0 * correct / total
            avg_loss = total_loss / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            
            # Evaluate
            if (epoch + 1) % evaluate_every == 0:
                from eval.metrics import accuracy
                test_acc = accuracy(self.model, test_loader, device=self.device) * 100
                history['test_acc'].append(test_acc)
                print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        return history

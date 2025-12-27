import torch
from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES, CLASS_NAMES
from trainers.trainer import train_one_task
from eval.metrics import accuracy, per_class_accuracy
from replay.buffer import ReplayBuffer

class TrueContinualTrainer:
    def __init__(self, model, use_replay = True, device = "cuda", num_tasks = 5):
        """
        Continual Learning trainer with Experience Replay
        
        Args:
            model: Neural network model with 10 output classes
            use_replay: Use Experience Replay (ER) - recommended for TRUE CL
            device: "cuda" or "cpu"
            num_tasks: Number of tasks (5 for Fashion-MNIST)
        
        Buffer auto-scales: 500 samples/class × 10 classes = 5000 total capacity
        """
        self.model = model
        self.use_replay = use_replay
        self.device = device
        self.num_tasks = num_tasks

        # Experience replay buffer (Auto-scaling: 500 samples/class)
        self.replay_buffer = ReplayBuffer(samples_per_class=500, max_classes=10) if use_replay else None

        # Track result
        self.task_acc_history = []
        self.per_class_acc_history = []  # NEW: Track per-class accuracies
        self.all_test_loaders = []
        self.all_val_loaders = []  # NEW: Track validation loaders
    
    def train_all_tasks(self, epochs_per_task = 10, batch_size = 128, lr = 0.01, data_root = "./data"):
        method_name = "Experience Replay" if self.use_replay else "Finetune (No CL)"
        print(f"\n{'-'*60}")
        print(f"TRUE Continual Learning: {method_name}")
        print(f"Training ONLY on new classes per task")
        if self.use_replay and self.replay_buffer:
            print(f"Replay Buffer: Auto-scaling (500 samples/class × 10 classes = {self.replay_buffer.total_size} total)")
        print(f"{'-'*60}\n")

        for task_id in range(self.num_tasks):
            print(f"\n{'-'*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'-'*60}")

            # Get data for current task only (70% train, 30% val)
            train_loader, val_loader, test_loader, classes = get_task_loaders_true_continual(
                task_id=task_id,
                batch_size=batch_size,
                root=data_root,
                train_ratio=0.7  # 70% train, 30% validation
            )
            self.all_test_loaders.append(test_loader)
            self.all_val_loaders.append(val_loader)
            # ER automatically mixes old samples during training
            self.model = train_one_task(
                self.model,
                train_loader,
                val_loader,  # Use val_loader for validation during training
                device = self.device,
                epochs = epochs_per_task,
                lr = lr,
                replay_buffer = self.replay_buffer,
            )

            # ═══════════════════════════════════════════════════════════════
            # POST-TASK EVALUATION: Validation set (30% of current task data)
            # ═══════════════════════════════════════════════════════════════
            print(f"\n{'═'*70}")
            print(f"VALIDATION TESTING LOOP - After Task {task_id} Training Complete")
            print(f"{'═'*70}")
            
            # Test on ALL validation sets (current + all previous)
            print(f"\n[Testing on ALL Tasks - Validation Set (30% split)]")
            print(f"{'─'*70}")
            
            for test_task_id in range(task_id + 1):
                test_val_loader = self.all_val_loaders[test_task_id]
                test_classes = get_task_loaders_true_continual(test_task_id, batch_size, data_root, train_ratio=0.7)[3]
                
                # Overall accuracy for this task
                task_acc = accuracy(self.model, test_val_loader, device=self.device)
                
                # Per-class accuracy
                per_class = per_class_accuracy(self.model, test_val_loader, device=self.device, num_classes=10)
                
                status = "✓ CURRENT" if test_task_id == task_id else "  Previous"
                print(f"\n{status} Task {test_task_id} ({TASK_THEMES[test_task_id]}): {task_acc*100:.2f}%")
                
                for class_id in test_classes:
                    class_name = CLASS_NAMES[class_id]
                    acc = per_class[class_id]
                    
                    # Visual indicator for accuracy level
                    if acc >= 0.90:
                        indicator = "🟢"
                    elif acc >= 0.80:
                        indicator = "🟡"
                    elif acc >= 0.70:
                        indicator = "🟠"
                    else:
                        indicator = "🔴"
                    
                    # Progress bar
                    bar_length = int(acc * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    
                    print(f"    {indicator} Class {class_id} ({class_name:15s}): [{bar}] {acc*100:5.1f}%")
            
            print(f"{'─'*70}")
            
            # ═══════════════════════════════════════════════════════════════
            # REPLAY BUFFER ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            if self.replay_buffer:
                self.replay_buffer.analyze_buffer(class_names=CLASS_NAMES)
                
                # Show buffer effectiveness
                print(f"{'═'*70}")
                print(f"BUFFER EFFECTIVENESS PROOF")
                print(f"{'═'*70}\n")
                
                if task_id > 0:
                    # Compare current vs initial accuracy for old tasks
                    print(f"[Forgetting Analysis]")
                    for prev_task_id in range(task_id):
                        initial_acc = self.task_acc_history[prev_task_id][prev_task_id]
                        current_acc = accuracy(self.model, self.all_val_loaders[prev_task_id], device=self.device)
                        forgetting = (initial_acc - current_acc) * 100
                        
                        if forgetting < 5:
                            status = "✓ Excellent retention"
                        elif forgetting < 10:
                            status = "○ Good retention"
                        elif forgetting < 15:
                            status = "△ Moderate forgetting"
                        else:
                            status = "✗ Significant forgetting"
                        
                        print(f"  Task {prev_task_id}: {initial_acc*100:.1f}% → {current_acc*100:.1f}% "
                              f"(Δ {forgetting:+.1f}%) {status}")
                    
                    avg_forgetting = sum(
                        (self.task_acc_history[i][i] - accuracy(self.model, self.all_val_loaders[i], device=self.device)) * 100
                        for i in range(task_id)
                    ) / task_id
                    
                    print(f"\n  Average forgetting: {avg_forgetting:.2f}%")
                    print(f"  Buffer effectiveness: {100 - avg_forgetting:.1f}% knowledge retained")
                    
                    if avg_forgetting < 10:
                        print(f"  ✓ Buffer is HIGHLY EFFECTIVE at preventing forgetting!")
                    elif avg_forgetting < 20:
                        print(f"  ○ Buffer is MODERATELY EFFECTIVE")
                    else:
                        print(f"  △ Buffer shows LIMITED effectiveness")
                
                print(f"\n{'═'*70}\n")
            
            # Evaluate on ALL test sets (for tracking overall performance)
            task_accuracies = [0.0] * self.num_tasks
            all_per_class = {}
            
            for i in range(task_id + 1):
                acc = accuracy(self.model, self.all_test_loaders[i], device = self.device)
                task_accuracies[i] = acc
                
                # Get per-class accuracies for test set
                test_per_class = per_class_accuracy(self.model, self.all_test_loaders[i], device=self.device, num_classes=10)
                all_per_class[i] = test_per_class

            print(f"[INFO] Task {task_id} training complete: {task_accuracies[task_id]*100:.2f}%")
            if task_id > 0:
                prev_accs = ', '.join([f'{acc*100:.1f}%' for acc in task_accuracies[:task_id]])
                print(f"[INFO] Previous tasks: [{prev_accs}]")
            
            self.task_acc_history.append(task_accuracies)
            self.per_class_acc_history.append(all_per_class)

        # Final evaluation
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        final_accuracies = self.task_acc_history[-1]
        for i in range(len(self.all_test_loaders)):
            print(f"  Task {i} ({TASK_THEMES.get(i, f'Task {i}')}): {final_accuracies[i]*100:.2f}%")
        
        self._print_results(final_accuracies)
        
        return self.task_acc_history
    
    def _print_results(self, final_accuracies):
        """Print final results"""
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        
        avg_acc = sum(final_accuracies) / len(final_accuracies)
        print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")
        
        if avg_acc >= 0.90:
            print(f"  Status: EXCELLENT - Achieved >90% accuracy")
        elif avg_acc >= 0.80:
            print(f"  Status: GOOD - High accuracy maintained")
        elif avg_acc >= 0.70:
            print(f"  Status: DECENT - Room for improvement")
        else:
            print(f"  Status: NEEDS WORK - Significant forgetting")
        
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
            "accuracy_matrix": self.task_acc_history
        }
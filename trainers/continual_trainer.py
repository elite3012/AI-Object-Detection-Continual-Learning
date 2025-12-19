import torch
from data.fashion_mnist_true_continual import get_task_loaders_true_continual, TASK_THEMES
from trainers.trainer import train_one_task
from eval.metrics import accuracy
from replay.buffer import ReplayBuffer

class TrueContinualTrainer:
    def __init__(self, model, use_replay = True, device = "cuda", num_tasks = 5, buffer_size = 500):
        """
        Continual Learning trainer with Experience Replay
        
        Args:
            model: Neural network model with 10 output classes
            use_replay: Use Experience Replay (ER) - recommended for TRUE CL
            device: "cuda" or "cpu"
            num_tasks: Number of tasks (5 for Fashion-MNIST)
            buffer_size: Samples per class in replay buffer (default: 500)
        """
        self.model = model
        self.use_replay = use_replay
        self.device = device
        self.num_tasks = num_tasks

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(m_per_class = buffer_size) if use_replay else None

        # Track result
        self.task_acc_history = []
        self.all_test_loaders = []
    
    def train_all_tasks(self, epochs_per_task = 10, batch_size = 128, lr = 0.01, data_root = "./data"):
        method_name = "Experience Replay" if self.use_replay else "Finetune (No CL)"
        print(f"\n{'-'*60}")
        print(f"TRUE Continual Learning: {method_name}")
        print(f"Training ONLY on new classes per task")
        if self.use_replay and self.replay_buffer:
            print(f"Replay Buffer: {self.replay_buffer.m_per_class} samples/class")
        print(f"{'-'*60}\n")

        for task_id in range(self.num_tasks):
            print(f"\n{'-'*60}")
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print(f"{'-'*60}")

            # Get data for current task only
            train_loader, test_loader, classes = get_task_loaders_true_continual(task_id = task_id,
                                                                                 batch_size = batch_size,
                                                                                 root = data_root)
            self.all_test_loaders.append(test_loader)
            # ER automatically mixes old samples during training
            self.model = train_one_task(
                self.model,
                train_loader,
                test_loader,
                device = self.device,
                epochs = epochs_per_task,
                lr = lr,
                replay_buffer = self.replay_buffer,
            )

            # Evaluate on ALL tasks seen so far
            task_accuracies = [0.0] * self.num_tasks
            for i in range(task_id + 1):
                acc = accuracy(self.model, self.all_test_loaders[i], device = self.device)
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
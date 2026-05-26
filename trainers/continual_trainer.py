from data.fashion_mnist_true_continual import (
    CLASS_NAMES,
    TASKS,
    TASK_THEMES,
    get_task_loaders_true_continual,
)
from eval.metrics import accuracy, per_class_accuracy
from replay.buffer import ReplayBuffer
from trainers.trainer import train_one_task


class TrueContinualTrainer:
    def __init__(self, model, use_replay=True, device="cuda", num_tasks=5):
        """
        Continual learning trainer with optional experience replay.

        Each task introduces only two new Fashion-MNIST classes. When replay is
        enabled, previous examples are mixed back into training from a fixed-size
        replay buffer.
        """
        self.model = model
        self.use_replay = use_replay
        self.device = device
        self.num_tasks = num_tasks

        self.replay_buffer = (
            ReplayBuffer(samples_per_class=500, max_classes=10)
            if use_replay
            else None
        )

        self.task_acc_history = []
        self.per_class_acc_history = []
        self.all_test_loaders = []
        self.all_val_loaders = []

    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.01, data_root="./data"):
        method_name = "Experience Replay" if self.use_replay else "Finetune (No Replay)"

        print("\n" + "-" * 60)
        print(f"TRUE Continual Learning: {method_name}")
        print("Training only on new classes per task")
        if self.replay_buffer:
            print(
                "Replay Buffer: fixed budget "
                f"{self.replay_buffer.total_size} samples "
                f"({self.replay_buffer.samples_per_class}/class at full capacity)"
            )
        print("-" * 60 + "\n")

        for task_id in range(self.num_tasks):
            print("\n" + "-" * 60)
            print(f"TASK {task_id}: {TASK_THEMES.get(task_id, f'Task {task_id}')}")
            print("-" * 60)

            train_loader, val_loader, test_loader, _ = get_task_loaders_true_continual(
                task_id=task_id,
                batch_size=batch_size,
                root=data_root,
                train_ratio=0.7,
            )
            self.all_val_loaders.append(val_loader)
            self.all_test_loaders.append(test_loader)

            self.model = train_one_task(
                self.model,
                train_loader,
                val_loader,
                device=self.device,
                epochs=epochs_per_task,
                lr=lr,
                replay_buffer=self.replay_buffer,
                task_id=task_id,
            )

            print("\n" + "=" * 70)
            print(f"VALIDATION LOOP - After Task {task_id} Training")
            print("=" * 70)
            print("\n[Testing on all learned validation splits]")
            print("-" * 70)

            for test_task_id in range(task_id + 1):
                test_val_loader = self.all_val_loaders[test_task_id]
                task_acc = accuracy(self.model, test_val_loader, device=self.device)
                per_class = per_class_accuracy(
                    self.model,
                    test_val_loader,
                    device=self.device,
                    num_classes=10,
                )

                status = "[current]" if test_task_id == task_id else "[previous]"
                print(
                    f"\n{status} Task {test_task_id} "
                    f"({TASK_THEMES[test_task_id]}): {task_acc * 100:.2f}%"
                )

                for class_id in TASKS[test_task_id]:
                    class_name = CLASS_NAMES[class_id]
                    class_acc = per_class[class_id]
                    if class_acc >= 0.90:
                        indicator = "high"
                    elif class_acc >= 0.80:
                        indicator = "good"
                    elif class_acc >= 0.70:
                        indicator = "fair"
                    else:
                        indicator = "low"

                    bar_length = int(class_acc * 40)
                    bar = "#" * bar_length + "." * (40 - bar_length)
                    print(
                        f"    {indicator:4s} Class {class_id} ({class_name:15s}): "
                        f"[{bar}] {class_acc * 100:5.1f}%"
                    )

            print("-" * 70)

            if self.replay_buffer:
                self.replay_buffer.analyze_buffer(class_names=CLASS_NAMES)
                self._print_forgetting_probe(task_id)

            task_accuracies = [0.0] * self.num_tasks
            all_per_class = {}
            for test_task_id in range(task_id + 1):
                task_accuracies[test_task_id] = accuracy(
                    self.model,
                    self.all_test_loaders[test_task_id],
                    device=self.device,
                )
                all_per_class[test_task_id] = per_class_accuracy(
                    self.model,
                    self.all_test_loaders[test_task_id],
                    device=self.device,
                    num_classes=10,
                )

            print(
                f"[INFO] Task {task_id} complete: "
                f"{task_accuracies[task_id] * 100:.2f}% test accuracy"
            )
            if task_id > 0:
                previous = ", ".join(
                    f"{acc * 100:.1f}%" for acc in task_accuracies[:task_id]
                )
                print(f"[INFO] Previous tasks: [{previous}]")

            self.task_acc_history.append(task_accuracies)
            self.per_class_acc_history.append(all_per_class)

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_accuracies = self.task_acc_history[-1]
        for task_id in range(len(self.all_test_loaders)):
            print(
                f"  Task {task_id} ({TASK_THEMES.get(task_id, f'Task {task_id}')}): "
                f"{final_accuracies[task_id] * 100:.2f}%"
            )

        self._print_results(final_accuracies)
        return self.task_acc_history

    def _print_forgetting_probe(self, task_id):
        print("=" * 70)
        print("BUFFER EFFECTIVENESS PROBE")
        print("=" * 70 + "\n")

        if task_id == 0:
            print("  Need at least two tasks before forgetting can be estimated.\n")
            return

        print("[Forgetting Analysis]")
        forgetting_values = []
        for prev_task_id in range(task_id):
            initial_acc = self.task_acc_history[prev_task_id][prev_task_id]
            current_acc = accuracy(
                self.model,
                self.all_val_loaders[prev_task_id],
                device=self.device,
            )
            forgetting = (initial_acc - current_acc) * 100
            forgetting_values.append(forgetting)

            if forgetting < 5:
                status = "Excellent retention"
            elif forgetting < 10:
                status = "Good retention"
            elif forgetting < 15:
                status = "Moderate forgetting"
            else:
                status = "Significant forgetting"

            print(
                f"  Task {prev_task_id}: {initial_acc * 100:.1f}% -> "
                f"{current_acc * 100:.1f}% (delta {forgetting:+.1f}%) {status}"
            )

        avg_forgetting = sum(forgetting_values) / len(forgetting_values)
        print(f"\n  Average forgetting: {avg_forgetting:.2f}%")
        print(f"  Retention proxy: {100 - avg_forgetting:.1f}%")

        if avg_forgetting < 10:
            print("  Buffer is highly effective at preventing forgetting.")
        elif avg_forgetting < 20:
            print("  Buffer is moderately effective.")
        else:
            print("  Buffer shows limited effectiveness.")
        print("=" * 70 + "\n")

    def _print_results(self, final_accuracies):
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        avg_acc = sum(final_accuracies) / len(final_accuracies)
        print(f"\n  Average Accuracy: {avg_acc * 100:.2f}%")

        if avg_acc >= 0.90:
            print("  Status: EXCELLENT - Achieved >90% accuracy")
        elif avg_acc >= 0.80:
            print("  Status: GOOD - High accuracy maintained")
        elif avg_acc >= 0.70:
            print("  Status: DECENT - Room for improvement")
        else:
            print("  Status: NEEDS WORK - Significant forgetting")

        print("\n" + "=" * 60 + "\n")

    def get_metrics(self):
        """Compute aggregate continual-learning metrics."""
        if not self.task_acc_history:
            return {}

        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)

        bwt = 0.0
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                bwt += self.task_acc_history[-1][i] - self.task_acc_history[i][i]
            bwt /= self.num_tasks - 1

        forgetting = 0.0
        if self.num_tasks > 1:
            for i in range(self.num_tasks - 1):
                max_acc = max(self.task_acc_history[j][i] for j in range(i, self.num_tasks))
                current_acc = self.task_acc_history[-1][i]
                forgetting += max_acc - current_acc
            forgetting /= self.num_tasks - 1

        return {
            "average_accuracy": avg_acc,
            "backward_transfer": bwt,
            "forgetting": forgetting,
            "final_accuracies": final_accs,
            "accuracy_matrix": self.task_acc_history,
        }

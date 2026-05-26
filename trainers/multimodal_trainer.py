import torch
from torch import nn, optim

from data.fashion_mnist_true_continual import (
    CLASS_NAMES,
    TASKS,
    TASK_THEMES,
    get_task_loaders_true_continual,
)
from data.fashion_text import get_text_description
from models.text_encoder import encode_texts
from replay.buffer import ReplayBuffer


class MultiModalContinualTrainer:
    """
    Multi-modal continual learning with vision features, text descriptions, and
    optional experience replay.
    """

    def __init__(
        self,
        multimodal_model,
        use_replay=True,
        device="cuda",
        num_tasks=5,
        text_mode="simple",
        text_dropout=0.3,
    ):
        self.model = multimodal_model
        self.use_replay = use_replay
        self.device = device
        self.num_tasks = num_tasks
        self.text_mode = text_mode
        self.text_dropout = text_dropout

        self.replay_buffer = (
            ReplayBuffer(samples_per_class=500, max_classes=10)
            if use_replay
            else None
        )

        self.task_acc_history = []
        self.per_class_acc_history = []
        self.all_test_loaders = []
        self.all_val_loaders = []

    def train_one_task(self, train_loader, test_loader, epochs=10, lr=0.0005, callback=None, task_id=0):
        """Train the multi-modal model on one task."""
        self.model.to(self.device)
        self.model.train()

        vision_params = list(self.model.vision_encoder.parameters())
        text_params = list(self.model.text_encoder.parameters())
        fusion_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())

        optimizer = optim.AdamW(
            [
                {"params": vision_params, "lr": lr, "weight_decay": 0.01},
                {"params": text_params, "lr": lr * 2, "weight_decay": 0.01},
                {"params": fusion_params, "lr": lr * 1.5, "weight_decay": 0.01},
            ]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.1,
        )
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            if callback:
                callback.on_epoch_start(task_id, epoch - 1)

            total_loss = 0.0
            n_batches = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                texts = [get_text_description(int(label), mode=self.text_mode) for label in labels]
                input_ids, attention_mask = encode_texts(texts, device=self.device)

                if self.replay_buffer is not None:
                    self.replay_buffer.add_batch(images, labels)
                    replay_images, replay_labels = self.replay_buffer.sample(int(images.size(0) * 0.7))
                    if replay_images is not None:
                        replay_images = replay_images.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_texts = [
                            get_text_description(int(label), mode=self.text_mode)
                            for label in replay_labels
                        ]
                        replay_input_ids, replay_attention_mask = encode_texts(
                            replay_texts,
                            device=self.device,
                        )

                        images = torch.cat([images, replay_images])
                        labels = torch.cat([labels, replay_labels])
                        input_ids = torch.cat([input_ids, replay_input_ids])
                        attention_mask = torch.cat([attention_mask, replay_attention_mask])

                logits = self.model(images, input_ids, attention_mask)
                loss = loss_fn(logits, labels)

                if torch.rand(1).item() < self.text_dropout:
                    vision_logits = self.model.forward_vision_only(images)
                    vision_loss = loss_fn(vision_logits, labels)
                    loss = 0.7 * loss + 0.3 * vision_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if callback:
                    callback.on_batch_end(
                        task_id,
                        epoch - 1,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                    )

            avg_loss = total_loss / max(n_batches, 1)
            if test_loader is not None:
                acc = self.evaluate_task(test_loader)
                print(f"[LOG] Epoch {epoch}: loss = {avg_loss:.4f}, acc = {acc:.4f}")
            else:
                print(f"[LOG] Epoch {epoch}: loss = {avg_loss:.4f}")

            if callback:
                callback.on_epoch_end(task_id, epoch - 1, avg_loss)

            scheduler.step()

    def evaluate_task(self, test_loader):
        """Evaluate on a single task."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                texts = [get_text_description(int(label), mode=self.text_mode) for label in labels]
                input_ids, attention_mask = encode_texts(texts, device=self.device)

                logits = self.model(images, input_ids, attention_mask)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()
        return correct / max(total, 1)

    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.0005, data_root="./data"):
        method_name = "Multi-Modal + Experience Replay" if self.use_replay else "Multi-Modal + Finetune"

        print("\n" + "-" * 60)
        print(f"Multi-Modal Continual Learning: {method_name}")
        print("Training with vision and text modalities")
        if self.replay_buffer:
            print(
                "Replay Buffer: fixed budget "
                f"{self.replay_buffer.total_size} samples "
                f"({self.replay_buffer.samples_per_class}/class at full capacity)"
            )
        print(f"Text Mode: {self.text_mode}")
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

            self.train_one_task(
                train_loader,
                val_loader,
                epochs=epochs_per_task,
                lr=lr,
                task_id=task_id,
            )

            print("\n" + "=" * 70)
            print(f"VALIDATION LOOP - After Task {task_id} Training")
            print("=" * 70)
            print("\n[Testing on all learned validation splits]")
            print("-" * 70)

            for test_task_id in range(task_id + 1):
                task_acc = self.evaluate_task(self.all_val_loaders[test_task_id])
                per_class = self.evaluate_per_class(self.all_val_loaders[test_task_id])
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
                task_accuracies[test_task_id] = self.evaluate_task(self.all_test_loaders[test_task_id])
                all_per_class[test_task_id] = self.evaluate_per_class(self.all_test_loaders[test_task_id])

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

        self._print_final_results()
        return self.task_acc_history

    def _print_forgetting_probe(self, task_id):
        print("=" * 70)
        print("BUFFER EFFECTIVENESS PROBE")
        print("=" * 70 + "\n")

        if task_id == 0:
            print("  Need at least two tasks before forgetting can be estimated.\n")
            return

        forgetting_values = []
        for prev_task_id in range(task_id):
            initial_acc = self.task_acc_history[prev_task_id][prev_task_id]
            current_acc = self.evaluate_task(self.all_val_loaders[prev_task_id])
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
        print("=" * 70 + "\n")

    def _print_final_results(self):
        """Print final results."""
        print("\n" + "=" * 60)
        print("FINAL MULTI-MODAL RESULTS")
        print("=" * 60)

        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)

        for task_id, acc in enumerate(final_accs):
            print(f"  Task {task_id} ({TASK_THEMES.get(task_id, f'Task {task_id}')}): {acc * 100:.2f}%")

        print(f"\n  Average Accuracy: {avg_acc * 100:.2f}%")

        if self.num_tasks > 1:
            forgetting = 0.0
            for task_id in range(self.num_tasks - 1):
                max_acc = max(self.task_acc_history[j][task_id] for j in range(task_id, self.num_tasks))
                forgetting += max_acc - final_accs[task_id]
            forgetting /= self.num_tasks - 1
            print(f"  Forgetting: {forgetting * 100:.2f}%")

        if avg_acc >= 0.90:
            status = "EXCELLENT - Multi-modal synergy working"
        elif avg_acc >= 0.80:
            status = "GOOD - Strong multi-modal learning"
        elif avg_acc >= 0.70:
            status = "DECENT - Room for improvement"
        else:
            status = "NEEDS WORK - Check fusion strategy"

        print(f"  Status: {status}")
        print("\n" + "=" * 60 + "\n")

    def get_metrics(self):
        """Get evaluation metrics."""
        if not self.task_acc_history:
            return {}

        final_accs = self.task_acc_history[-1]
        avg_acc = sum(final_accs) / len(final_accs)

        forgetting = 0.0
        if self.num_tasks > 1:
            for task_id in range(self.num_tasks - 1):
                max_acc = max(self.task_acc_history[j][task_id] for j in range(task_id, self.num_tasks))
                current_acc = self.task_acc_history[-1][task_id]
                forgetting += max_acc - current_acc
            forgetting /= self.num_tasks - 1

        return {
            "average_accuracy": avg_acc,
            "forgetting": forgetting,
            "final_accuracies": final_accs,
            "accuracy_matrix": self.task_acc_history,
        }

    def evaluate_per_class(self, test_loader):
        """Evaluate per-class accuracy on a test set."""
        self.model.eval()
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                texts = [get_text_description(int(label), mode=self.text_mode) for label in labels]
                input_ids, attention_mask = encode_texts(texts, device=self.device)

                logits = self.model(images, input_ids, attention_mask)
                pred = logits.argmax(1)

                for label, prediction in zip(labels, pred):
                    label_item = label.item()
                    class_total[label_item] += 1
                    if prediction == label:
                        class_correct[label_item] += 1

        return {
            class_id: class_correct[class_id] / class_total[class_id]
            if class_total[class_id] > 0
            else 0.0
            for class_id in range(10)
        }

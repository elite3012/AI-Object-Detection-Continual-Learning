"""
PEFT Continual Learning Trainer
Combines TRUE Continual Learning with Parameter-Efficient Fine-Tuning (LoRA)
"""
import torch
from trainers.continual_trainer import TrueContinualTrainer
from models.peft_lora import apply_lora_to_model, get_lora_parameters
from models.simple_cnn_multiclass import SimpleCNNMulticlass

class PEFTContinualTrainer(TrueContinualTrainer):
    """
    TRUE Continual Learning with PEFT (LoRA)
    Only trains LoRA adapters (1-10% of parameters) and optionally unfreezes backbone layers.
    """
    def __init__(self, model, use_replay=True, device="cuda", num_tasks=5,
                 lora_rank=24, lora_alpha=48, lora_dropout=0.0,
                 unfreeze_backbone=False, num_classes=None):
        """
        PEFT Continual Learning trainer
        
        Args:
            model: Foundation model (CNN, ViT, ResNet, etc.)
            use_replay: Use Experience Replay
            device: "cuda" or "cpu"
            num_tasks: Number of tasks
            buffer_size: Samples per class in replay buffer (default: 500)
            lora_rank: LoRA rank (16-32, higher=better capacity for hard tasks)
            lora_alpha: LoRA scaling factor (32-64, 2x rank is optimal)
            lora_dropout: Dropout for LoRA layers (0.0 = no dropout)
            unfreeze_backbone: Whether to unfreeze backbone layers for partial updates
            num_classes: Number of output classes (required if model does not define it)
        """
        # Apply LoRA before calling parent __init__
        print(f"\n{'='*60}")
        print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
        print(f"Unfreeze Backbone: {unfreeze_backbone}")
        print(f"{'='*60}")
        
        model, trainable_params, total_params = apply_lora_to_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=None,  # Apply to all Linear/Conv2d layers
            dropout=lora_dropout
        )

        # Ensure num_classes is provided
        if num_classes is None:
            raise ValueError("num_classes must be provided if the model does not define it.")

        model = SimpleCNNMulticlass(num_classes=num_classes, unfreeze_backbone=unfreeze_backbone)

        self.trainable_params = trainable_params
        self.total_params = total_params
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Call parent constructor with LoRA-enhanced model
        super().__init__(
            model=model,
            use_replay=use_replay,
            device=device,
            num_tasks=num_tasks,
            buffer_size=buffer_size
        )
    
    def train_all_tasks(self, epochs_per_task=10, batch_size=128, lr=0.002, data_root="./data"):
        """
        Train on all tasks with PEFT
        
        Note: LoRA uses higher lr (0.002 default, 4x higher than full fine-tuning's 0.0005)
        because we're only updating 4% of parameters, allowing faster convergence
        """
        method_name = "LoRA + Experience Replay" if self.use_replay else "LoRA + Finetune"
        
        print(f"\n{'-'*60}")
        print(f"PEFT Continual Learning: {method_name}")
        print(f"Training ONLY on new classes per task")
        if self.use_replay and self.replay_buffer:
            print(f"Replay Buffer: {self.replay_buffer.m_per_class} samples/class")
        print(f"LoRA Config: rank={self.lora_rank}, alpha={self.lora_alpha}")
        print(f"LoRA LR: {lr} (optimized for parameter-efficient training)")
        print(f"Trainable: {self.trainable_params:,} / {self.total_params:,} params ({self.trainable_params/self.total_params*100:.2f}%)")
        print(f"{'-'*60}\n")
        
        # Call parent's train_all_tasks with LoRA-optimized learning rate
        history = super().train_all_tasks(
            epochs_per_task=epochs_per_task,
            batch_size=batch_size,
            lr=lr,
            data_root=data_root
        )
        
        return history
    
    def get_lora_state_dict(self):
        """
        Get only LoRA parameters state dict (for saving)
        Much smaller than full model state dict
        """
        lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                lora_state[name] = param.cpu().detach()
        return lora_state
    
    def save_lora_weights(self, path):
        """Save only LoRA weights (very small file)"""
        lora_state = self.get_lora_state_dict()
        torch.save(lora_state, path)
        size_mb = sum(p.numel() * 4 for p in lora_state.values()) / 1e6
        print(f"[LoRA] Saved to {path} ({size_mb:.2f} MB)")
    
    def load_lora_weights(self, path):
        """Load LoRA weights"""
        lora_state = torch.load(path)
        self.model.load_state_dict(lora_state, strict=False)
        print(f"[LoRA] Loaded from {path}")

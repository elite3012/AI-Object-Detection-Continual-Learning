"""
Hardware-aware optimizer.

Selects a compression strategy for a target deployment profile by combining
pruning and quantization utilities.
"""

from .pruning import count_nonzero_parameters, count_parameters, get_model_size_mb, prune_model
from .quantization import convert_to_fp16, quantize_model


class HardwareOptimizer:
    """Auto-select optimization based on simple hardware constraints."""

    PRESETS = {
        "mobile": {
            "quantization": "int8",
            "sparsity": 0.5,
            "target_size_mb": 5.0,
            "priority": "size",
            "description": "Mobile deployment (ARM CPU, limited memory)",
        },
        "gpu": {
            "quantization": "fp16",
            "sparsity": 0.3,
            "target_size_mb": 50.0,
            "priority": "speed",
            "description": "GPU deployment (NVIDIA GPU, high throughput)",
        },
        "edge": {
            "quantization": "int8",
            "sparsity": 0.7,
            "target_size_mb": 2.0,
            "priority": "size",
            "description": "Edge device (Raspberry Pi, IoT)",
        },
        "cloud": {
            "quantization": "fp16",
            "sparsity": 0.2,
            "target_size_mb": 100.0,
            "priority": "accuracy",
            "description": "Cloud deployment (server GPU, high accuracy)",
        },
        "custom": {
            "quantization": None,
            "sparsity": None,
            "target_size_mb": None,
            "priority": "balanced",
            "description": "Custom configuration",
        },
    }

    def __init__(self, preset="mobile"):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")

        self.preset = preset
        self.config = self.PRESETS[preset].copy()

        print("\n" + "=" * 70)
        print(f"Hardware Optimizer: {preset.upper()}")
        print(f"  {self.config['description']}")
        print(f"  Quantization: {self.config['quantization']}")
        sparsity = self.config["sparsity"]
        print(f"  Sparsity: {sparsity * 100 if sparsity else 'N/A'}%")
        print(f"  Target size: {self.config['target_size_mb']} MB")
        print(f"  Priority: {self.config['priority']}")
        print("=" * 70 + "\n")

    def optimize(self, model, train_loader=None, device="cpu"):
        """Apply the configured compression strategy."""
        print("\n" + "=" * 70)
        print("Starting Hardware Optimization")
        print("=" * 70 + "\n")

        optimized_model = model
        all_metrics = {"preset": self.preset, "steps": []}

        if self.config["sparsity"] and self.config["sparsity"] > 0:
            print(f"[Step 1] Pruning with sparsity={self.config['sparsity'] * 100:.0f}%")
            prune_result = prune_model(
                optimized_model,
                sparsity=self.config["sparsity"],
                method="magnitude",
                structured=True,
            )
            optimized_model = prune_result["pruned_model"]
            all_metrics["steps"].append(
                {"type": "pruning", "metrics": prune_result["metrics"]}
            )

        if self.config["quantization"]:
            print(f"[Step 2] Quantization to {self.config['quantization']}")

            if self.config["quantization"] == "int8":
                if train_loader is None:
                    print("  Warning: no train_loader provided. Skipping QAT.")
                else:
                    quant_result = quantize_model(
                        optimized_model,
                        train_loader,
                        dtype="qint8",
                        qat_epochs=3,
                        device=device,
                    )
                    optimized_model = quant_result["quantized_model"]
                    all_metrics["steps"].append(
                        {"type": "quantization_int8", "metrics": quant_result["metrics"]}
                    )

            elif self.config["quantization"] == "fp16":
                optimized_model, fp16_metrics = convert_to_fp16(optimized_model)
                all_metrics["steps"].append(
                    {"type": "quantization_fp16", "metrics": fp16_metrics}
                )

        final_metrics = self._calculate_final_metrics(model, optimized_model, all_metrics)

        print("\n" + "=" * 70)
        print("Hardware Optimization Complete")
        print(f"  Final compression: {final_metrics['total_compression_ratio']:.2f}x")
        print(f"  Final size: {final_metrics['final_size_mb']:.2f} MB")
        print(f"  Target achieved: {'yes' if final_metrics['target_achieved'] else 'no'}")
        print("=" * 70 + "\n")

        return {
            "optimized_model": optimized_model,
            "metrics": final_metrics,
            "strategy": self.config,
        }

    def _calculate_final_metrics(self, original_model, optimized_model, all_metrics):
        """Calculate overall compression metrics."""
        original_size = get_model_size_mb(original_model)
        final_size = get_model_size_mb(optimized_model)
        original_params = count_parameters(original_model)
        final_params = count_nonzero_parameters(optimized_model)
        target_size = self.config["target_size_mb"]

        return {
            "original_size_mb": original_size,
            "final_size_mb": final_size,
            "original_params": original_params,
            "final_params": final_params,
            "total_compression_ratio": original_size / final_size,
            "size_reduction_percent": (1 - final_size / original_size) * 100,
            "param_reduction_percent": (1 - final_params / original_params) * 100,
            "target_size_mb": target_size,
            "target_achieved": True if target_size is None else final_size <= target_size,
            "steps": all_metrics["steps"],
        }

    @staticmethod
    def list_presets():
        """List all available hardware presets."""
        print("\n" + "=" * 70)
        print("Available Hardware Presets")
        print("=" * 70 + "\n")

        for name, config in HardwareOptimizer.PRESETS.items():
            print(f"{name.upper():10s}: {config['description']}")
            print(f"            Quantization: {config['quantization']}")
            sparsity = config["sparsity"]
            print(f"            Sparsity: {sparsity * 100 if sparsity else 'N/A'}%")
            print(f"            Target size: {config['target_size_mb']} MB")
            print(f"            Priority: {config['priority']}")
            print()

        print("=" * 70 + "\n")

    def set_custom_config(self, quantization=None, sparsity=None, target_size_mb=None):
        """Set a custom optimization configuration."""
        if self.preset != "custom":
            print(f"Warning: changing preset from '{self.preset}' to 'custom'")
            self.preset = "custom"

        if quantization is not None:
            self.config["quantization"] = quantization

        if sparsity is not None:
            self.config["sparsity"] = sparsity

        if target_size_mb is not None:
            self.config["target_size_mb"] = target_size_mb

        print("\nCustom config updated:")
        print(f"  Quantization: {self.config['quantization']}")
        sparsity_value = self.config["sparsity"]
        print(f"  Sparsity: {sparsity_value * 100 if sparsity_value else 'N/A'}%")
        print(f"  Target size: {self.config['target_size_mb']} MB")


def auto_optimize(model, train_loader=None, target_hardware="mobile", device="cpu"):
    """Convenience wrapper for one-shot hardware optimization."""
    optimizer = HardwareOptimizer(preset=target_hardware)
    result = optimizer.optimize(model, train_loader, device)
    return result["optimized_model"]

"""
Hardware-Aware Model Optimizer
Automatically profiles models and applies best compression strategy
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import time
import psutil
import os
from .quantization import quantize_model, QuantizationConfig
from .pruning import prune_model, PruningConfig


class HardwareProfile:
    """Profile hardware capabilities"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu_count = os.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_name = None
            self.gpu_memory_gb = 0
    
    def __repr__(self):
        s = f"Device: {self.device}\n"
        s += f"CPU Cores: {self.cpu_count}\n"
        s += f"RAM: {self.ram_gb:.1f} GB\n"
        if self.gpu_name:
            s += f"GPU: {self.gpu_name}\n"
            s += f"GPU Memory: {self.gpu_memory_gb:.1f} GB"
        return s


class ModelProfiler:
    """
    Profile model performance metrics
    - Inference speed (FPS)
    - Memory usage
    - Model size
    - Accuracy
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def measure_inference_speed(
        self,
        model: nn.Module,
        input_shape: Tuple = (1, 1, 28, 28),
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Measure inference speed in FPS and latency
        
        Returns:
            dict with 'fps', 'latency_ms', 'throughput'
        """
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Synchronize GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        fps = num_iterations / elapsed
        latency_ms = (elapsed / num_iterations) * 1000
        
        # Batch throughput (images/sec with batch processing)
        batch_size = input_shape[0]
        throughput = fps * batch_size
        
        return {
            'fps': fps,
            'latency_ms': latency_ms,
            'throughput': throughput,
            'batch_size': batch_size
        }
    
    def measure_memory_usage(
        self,
        model: nn.Module,
        input_shape: Tuple = (1, 1, 28, 28)
    ) -> Dict:
        """
        Measure memory consumption
        
        Returns:
            dict with 'model_size_mb', 'peak_memory_mb'
        """
        model.to(self.device)
        
        # Model size
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_size_mb = buffer.tell() / (1024 * 1024)
        
        # Peak memory during inference
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
            dummy_input = torch.randn(input_shape).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # CPU memory harder to measure precisely
            peak_memory_mb = model_size_mb * 2  # Rough estimate
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': peak_memory_mb
        }
    
    def measure_accuracy(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> float:
        """Measure model accuracy on test set"""
        model.eval()
        model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels.to(self.device)).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def profile_model(
        self,
        model: nn.Module,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        input_shape: Tuple = (1, 1, 28, 28)
    ) -> Dict:
        """
        Complete model profiling
        
        Returns comprehensive metrics dict
        """
        print("\n[Profiling] Measuring model performance...")
        
        metrics = {}
        
        # Speed
        speed = self.measure_inference_speed(model, input_shape)
        metrics.update(speed)
        
        # Memory
        memory = self.measure_memory_usage(model, input_shape)
        metrics.update(memory)
        
        # Accuracy
        if test_loader:
            accuracy = self.measure_accuracy(model, test_loader)
            metrics['accuracy'] = accuracy
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['total_params'] = total_params
        metrics['trainable_params'] = trainable_params
        
        return metrics


class AutoOptimizer:
    """
    Automatically select and apply best optimization strategy
    Based on hardware constraints and target metrics
    """
    
    def __init__(
        self,
        target_metric: str = 'balanced',  # 'speed', 'size', 'accuracy', 'balanced'
        device: str = 'cpu'
    ):
        self.target_metric = target_metric
        self.device = device
        self.hw_profile = HardwareProfile()
        self.profiler = ModelProfiler(device)
    
    def recommend_strategy(self, model: nn.Module) -> Dict:
        """
        Recommend optimization strategy based on:
        1. Hardware capabilities
        2. Model characteristics
        3. Target metric
        
        Returns recommended config
        """
        print("\n" + "="*60)
        print("AUTO OPTIMIZER - ANALYZING MODEL")
        print("="*60)
        print(self.hw_profile)
        print("="*60)
        
        # Profile baseline model
        baseline = self.profiler.profile_model(model)
        
        print(f"\nBaseline Metrics:")
        print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
        print(f"  Params: {baseline['total_params']:,}")
        print(f"  FPS: {baseline['fps']:.1f}")
        print(f"  Latency: {baseline['latency_ms']:.2f} ms")
        
        # Decide strategy based on target and hardware
        recommendations = {}
        
        if self.target_metric == 'speed':
            # Prioritize inference speed
            if self.device == 'cuda':
                # GPU: FP16 quantization best for speed
                recommendations['quantization'] = 'float16'
                recommendations['pruning_sparsity'] = 0.3  # Light pruning
            else:
                # CPU: INT8 quantization
                recommendations['quantization'] = 'qint8'
                recommendations['pruning_sparsity'] = 0.5
        
        elif self.target_metric == 'size':
            # Prioritize model size
            recommendations['quantization'] = 'qint8'  # Best compression
            recommendations['pruning_sparsity'] = 0.7  # Aggressive pruning
        
        elif self.target_metric == 'accuracy':
            # Prioritize accuracy (light optimization)
            recommendations['quantization'] = 'float16'  # Minimal accuracy loss
            recommendations['pruning_sparsity'] = 0.2  # Very light pruning
        
        else:  # balanced
            # Balance all metrics
            recommendations['quantization'] = 'float16' if self.device == 'cuda' else 'qint8'
            recommendations['pruning_sparsity'] = 0.5
        
        print(f"\n✅ Recommended Strategy for '{self.target_metric}':")
        print(f"  Quantization: {recommendations['quantization']}")
        print(f"  Pruning Sparsity: {recommendations['pruning_sparsity']*100:.0f}%")
        
        return recommendations
    
    def optimize_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        auto_strategy: bool = True
    ) -> Dict:
        """
        Apply optimizations to model
        
        Args:
            model: Model to optimize
            train_loader: Training data for calibration/fine-tuning
            test_loader: Test data for accuracy measurement
            auto_strategy: Use automatic strategy recommendation
        
        Returns:
            dict with optimized models and comparison
        """
        print("\n" + "="*60)
        print("AUTO OPTIMIZATION PIPELINE")
        print("="*60)
        
        results = {}
        
        # Get strategy
        if auto_strategy:
            strategy = self.recommend_strategy(model)
        else:
            strategy = {
                'quantization': 'float16',
                'pruning_sparsity': 0.5
            }
        
        # Baseline metrics
        baseline_metrics = self.profiler.profile_model(model, test_loader)
        results['baseline'] = {
            'model': model,
            'metrics': baseline_metrics
        }
        
        # Apply Quantization
        print(f"\n{'='*60}")
        print("STEP 1: QUANTIZATION")
        print("="*60)
        
        quant_result = quantize_model(
            model,
            train_loader,
            dtype=strategy['quantization'],
            qat_epochs=2,
            device=self.device
        )
        
        quantized_model = quant_result['quantized_model']
        quant_metrics = self.profiler.profile_model(quantized_model, test_loader)
        
        results['quantized'] = {
            'model': quantized_model,
            'metrics': quant_metrics
        }
        
        # Apply Pruning
        print(f"\n{'='*60}")
        print("STEP 2: PRUNING")
        print("="*60)
        
        def fine_tune_fn(model, sparsity):
            """Quick fine-tuning after pruning"""
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break  # Just 1 batch for speed
        
        prune_result = prune_model(
            model,
            target_sparsity=strategy['pruning_sparsity'],
            method='channel',
            gradual=False,  # Skip gradual for speed
            train_fn=fine_tune_fn
        )
        
        pruned_model = prune_result['pruned_model']
        prune_metrics = self.profiler.profile_model(pruned_model, test_loader)
        
        results['pruned'] = {
            'model': pruned_model,
            'metrics': prune_metrics
        }
        
        # Combined: Prune + Quantize
        print(f"\n{'='*60}")
        print("STEP 3: COMBINED (PRUNE + QUANTIZE)")
        print("="*60)
        
        combined_result = quantize_model(
            pruned_model,
            train_loader,
            dtype=strategy['quantization'],
            qat_epochs=2,
            device=self.device
        )
        
        combined_model = combined_result['quantized_model']
        combined_metrics = self.profiler.profile_model(combined_model, test_loader)
        
        results['combined'] = {
            'model': combined_model,
            'metrics': combined_metrics
        }
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict):
        """Print comparison table"""
        print("\n" + "="*80)
        print("OPTIMIZATION COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'Size (MB)':<12} {'Params':<12} {'FPS':<10} {'Acc (%)':<10}")
        print("-"*80)
        
        for name, result in results.items():
            m = result['metrics']
            acc_str = f"{m.get('accuracy', 0):.2f}" if 'accuracy' in m else "N/A"
            
            print(f"{name.capitalize():<20} "
                  f"{m['model_size_mb']:<12.2f} "
                  f"{m['total_params']:<12,} "
                  f"{m['fps']:<10.1f} "
                  f"{acc_str:<10}")
        
        print("="*80)
        
        # Calculate improvements
        baseline = results['baseline']['metrics']
        combined = results['combined']['metrics']
        
        size_reduction = (1 - combined['model_size_mb'] / baseline['model_size_mb']) * 100
        speed_increase = (combined['fps'] / baseline['fps'] - 1) * 100
        param_reduction = (1 - combined['total_params'] / baseline['total_params']) * 100
        
        print(f"\n🚀 Combined Optimization Benefits:")
        print(f"   Size Reduction:   {size_reduction:.1f}%")
        print(f"   Speed Increase:   {speed_increase:.1f}%")
        print(f"   Param Reduction:  {param_reduction:.1f}%")
        
        if 'accuracy' in baseline and 'accuracy' in combined:
            acc_drop = baseline['accuracy'] - combined['accuracy']
            print(f"   Accuracy Drop:    {acc_drop:.2f}%")
        
        print("="*80)

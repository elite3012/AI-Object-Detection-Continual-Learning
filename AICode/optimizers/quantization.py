"""
Quantization-Aware Training (QAT) for Continual Learning
Supports INT8 and FP16 quantization with calibration
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from typing import Optional, Dict, Any
import copy


class QuantizationConfig:
    """Configuration for quantization"""
    
    def __init__(
        self,
        backend: str = 'qnnpack',  # 'fbgemm' for x86, 'qnnpack' for ARM
        qscheme: str = 'per_tensor_affine',  # or 'per_channel_affine'
        reduce_range: bool = False,
        dtype: str = 'qint8'  # 'qint8' or 'float16'
    ):
        self.backend = backend
        self.qscheme = qscheme
        self.reduce_range = reduce_range
        self.dtype = dtype
        
    def get_qconfig(self):
        """Get PyTorch qconfig"""
        if self.dtype == 'float16':
            return None  # FP16 uses different path
        
        if self.backend == 'fbgemm':
            return quant.get_default_qconfig('fbgemm')
        else:
            return quant.get_default_qconfig('qnnpack')


class QuantizableWrapper(nn.Module):
    """
    Wrapper that makes any model quantization-ready
    Adds QuantStub/DeQuantStub for quantization boundaries
    """
    
    def __init__(self, model: nn.Module, num_classes: int = 10):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
        self.num_classes = num_classes
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv+BN+ReLU for better quantization"""
        # Try to fuse common patterns
        for module in self.model.modules():
            if hasattr(module, 'fuse_model'):
                module.fuse_model()


class QuantizationAwareTrainer:
    """
    QAT trainer that calibrates and fine-tunes quantized models
    
    Workflow:
    1. Calibration: Collect activation statistics
    2. QAT Fine-tuning: Train with fake quantization
    3. Conversion: Convert to actual quantized model
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        device: str = 'cpu'
    ):
        self.original_model = model
        self.config = config
        self.device = device
        
        # For INT8 quantization
        if config.dtype == 'qint8':
            # Wrap model with quant stubs
            self.qat_model = QuantizableWrapper(copy.deepcopy(model))
            self.qat_model.to(device)
            
            # Set backend
            torch.backends.quantized.engine = config.backend
            
            # Configure quantization
            self.qat_model.qconfig = config.get_qconfig()
            
            # Prepare for QAT
            quant.prepare_qat(self.qat_model, inplace=True)
            
        # For FP16 quantization
        else:
            self.qat_model = copy.deepcopy(model)
            self.qat_model.half()  # Convert to FP16
            self.qat_model.to(device)
        
        self.quantized_model = None
        self.is_trained = False
    
    def calibrate(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        num_batches: int = 100
    ):
        """
        Calibration phase: collect activation statistics
        Only needed for PTQ (Post-Training Quantization)
        """
        if self.config.dtype == 'float16':
            return  # FP16 doesn't need calibration
        
        print(f"[Calibration] Processing {num_batches} batches...")
        self.qat_model.eval()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                
                images = images.to(self.device)
                if self.config.dtype == 'float16':
                    images = images.half()
                
                self.qat_model(images)
                
                if (i + 1) % 20 == 0:
                    print(f"  Batch {i+1}/{num_batches}")
        
        print("[Calibration] Done!")
    
    def train_qat(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
        lr: float = 0.0001
    ):
        """
        QAT fine-tuning with fake quantization
        This adapts the model to quantization noise
        """
        print(f"\n[QAT Training] Fine-tuning for {epochs} epochs...")
        
        self.qat_model.train()
        optimizer = torch.optim.Adam(self.qat_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.config.dtype == 'float16':
                    images = images.half()
                
                optimizer.zero_grad()
                outputs = self.qat_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            acc = 100.0 * correct / total
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        self.is_trained = True
        print("[QAT Training] Done!")
    
    def convert_to_quantized(self):
        """
        Convert QAT model to actual quantized model (INT8)
        This removes fake quantization and uses real quantized ops
        """
        if self.config.dtype == 'float16':
            # FP16 is already quantized
            self.quantized_model = self.qat_model
            return self.quantized_model
        
        if not self.is_trained:
            print("[Warning] QAT not trained yet, converting anyway...")
        
        print("\n[Conversion] Converting to INT8...")
        self.qat_model.eval()
        self.quantized_model = quant.convert(self.qat_model, inplace=False)
        print("[Conversion] Done!")
        
        return self.quantized_model
    
    def get_model_size(self, model: Optional[nn.Module] = None):
        """Calculate model size in MB"""
        if model is None:
            model = self.quantized_model if self.quantized_model else self.qat_model
        
        # Save to buffer and get size
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.tell() / (1024 * 1024)
        return size_mb
    
    def compare_models(self):
        """Compare original vs quantized model"""
        original_size = self.get_model_size(self.original_model)
        
        if self.quantized_model:
            quantized_size = self.get_model_size(self.quantized_model)
        else:
            quantized_size = self.get_model_size(self.qat_model)
        
        compression_ratio = original_size / quantized_size
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"Original Model:   {original_size:.2f} MB")
        print(f"Quantized Model:  {quantized_size:.2f} MB ({self.config.dtype})")
        print(f"Compression:      {compression_ratio:.2f}x smaller")
        print(f"Size Reduction:   {(1 - 1/compression_ratio)*100:.1f}%")
        print("=" * 60)
        
        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio
        }


def quantize_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    dtype: str = 'qint8',
    qat_epochs: int = 3,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    High-level API to quantize a model
    
    Args:
        model: Model to quantize
        train_loader: Data for calibration and QAT
        dtype: 'qint8' or 'float16'
        qat_epochs: Number of QAT fine-tuning epochs
        device: Device for training
    
    Returns:
        dict with 'quantized_model' and 'metrics'
    """
    print(f"\n{'='*60}")
    print(f"QUANTIZATION: {dtype.upper()}")
    print(f"{'='*60}")
    
    # Configure quantization
    config = QuantizationConfig(
        backend='qnnpack' if 'arm' in device.lower() or 'cpu' in device.lower() else 'fbgemm',
        dtype=dtype
    )
    
    # Create trainer
    trainer = QuantizationAwareTrainer(model, config, device)
    
    # Calibrate (for INT8)
    if dtype == 'qint8':
        trainer.calibrate(train_loader, num_batches=50)
    
    # QAT fine-tuning
    trainer.train_qat(train_loader, epochs=qat_epochs, lr=0.0001)
    
    # Convert to quantized
    quantized_model = trainer.convert_to_quantized()
    
    # Compare models
    metrics = trainer.compare_models()
    
    return {
        'quantized_model': quantized_model,
        'trainer': trainer,
        'metrics': metrics
    }


# FP16 Utilities
class FP16Wrapper(nn.Module):
    """Simple FP16 wrapper for inference"""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.half()
    
    def forward(self, x):
        return self.model(x.half()).float()


def convert_to_fp16(model: nn.Module) -> nn.Module:
    """Convert model to FP16"""
    return FP16Wrapper(copy.deepcopy(model))

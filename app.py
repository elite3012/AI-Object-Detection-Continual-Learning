"""
Streamlit Web UI with Real-Time Training Visualization
Enhanced for Report Screenshots
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from trainers.trainer import train_one_task
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import pandas as pd
import cv2
import os
import sys
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.simple_cnn_multiclass import SimpleCNNMulticlass
from models.peft_lora import apply_lora_to_model
from models.text_encoder import SimpleTextEncoder, get_tokenizer, encode_texts
from models.multimodal_fusion import MultiModalClassifier
from data.fashion_text import get_text_description
from trainers.continual_trainer import TrueContinualTrainer
from trainers.peft_trainer import PEFTContinualTrainer
from trainers.multimodal_trainer import MultiModalContinualTrainer
from trainers.hardware_trainer import HardwareContinualTrainer
from data.fashion_mnist_true_continual import CLASS_NAMES

# Page config
st.set_page_config(page_title="Continual Learning System", layout="wide")

# Session state
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'device' not in st.session_state:
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# Title
st.title("Continual Learning System")
st.markdown("**Fashion-MNIST Classification with Continual Learning Strategies**")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Device info
if torch.cuda.is_available():
    st.sidebar.info(f"Device: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Device: CPU")

# Training strategy selection
phase = st.sidebar.radio("Training Strategy", [
    "Experience Replay", 
    "PEFT/LoRA",
    "Multi-Modal (Vision + Text)",
    "Hardware Optimization"
])

# Common params
st.sidebar.subheader("Training Parameters")
num_tasks = st.sidebar.slider("Number of Tasks", 1, 5, 5)

# Phase-specific defaults for optimal performance
if "PEFT" in phase:
    # LoRA: Fewer epochs (faster convergence), larger batch size (less memory), higher LR
    epochs_per_task = st.sidebar.slider("Epochs per Task", 5, 30, 12)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 256)
    lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005], value=0.002)
    st.sidebar.info("LoRA uses larger batch size, fewer epochs, and higher LR for efficiency")
else:
    # Standard settings for other phases
    epochs_per_task = st.sidebar.slider("Epochs per Task", 5, 30, 15)
    batch_size = st.sidebar.slider("Batch Size", 32, 256, 128)
    lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

# Strategy-specific params
if "Experience Replay" == phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    if use_replay:
        st.sidebar.info("Buffer auto-scales: 500 samples/class × 10 classes = 5000 total")
elif "PEFT" in phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    if use_replay:
        st.sidebar.info("Buffer auto-scales: 500 samples/class × 10 classes = 5000 total")
    lora_rank = st.sidebar.slider("LoRA Rank", 8, 64, 24)
    lora_alpha = st.sidebar.slider("LoRA Alpha", 16, 128, 48)
elif "Multi-Modal" in phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    if use_replay:
        st.sidebar.info("Buffer auto-scales: 500 samples/class × 10 classes = 5000 total")
    fusion_type = st.sidebar.selectbox("Fusion Strategy", ["concat", "cross_attention", "gated"], index=0)
    text_mode = st.sidebar.selectbox("Text Mode", ["simple", "rich", "attributes"], index=0)
else:  # Hardware Optimization
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    if use_replay:
        st.sidebar.info("Buffer auto-scales: 500 samples/class × 10 classes = 5000 total")
    compression_strategy = st.sidebar.selectbox("Compression Strategy", ["end", "gradual", "per_task"], index=1)
    target_hardware = st.sidebar.selectbox("Target Hardware", ["mobile", "gpu", "edge"], index=0)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Training", "Testing", "History"])

# ===== TRAINING TAB =====
with tab1:
    st.header("Model Training")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Configuration")
        st.write(f"**Strategy:** {phase}")
        st.write(f"**Tasks:** {num_tasks}")
        st.write(f"**Epochs per Task:** {epochs_per_task}")
        st.write(f"**Batch Size:** {batch_size}")
        st.write(f"**Learning Rate:** {lr}")
    
    with col2:
        if st.session_state.trainer is not None:
            st.success("Status: Ready")
        else:
            st.info("Status: Not Trained")
    
    if st.button("Start Training", type="primary", use_container_width=True):
        # Create UI containers
        st.markdown("---")
        st.subheader("Training Progress")
        
        # Progress bars
        col_prog1, col_prog2, col_prog3 = st.columns(3)
        with col_prog1:
            st.write("**Overall Progress**")
            overall_progress = st.progress(0)
            progress_pct = st.empty()
        with col_prog2:
            st.write("**Current Task**")
            task_progress = st.progress(0)
            task_info = st.empty()
        with col_prog3:
            st.write("**Current Epoch**")
            epoch_progress = st.progress(0)
            epoch_info = st.empty()
        
        # Status cards
        st.markdown("###")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            task_status = st.empty()
        with col_stat2:
            epoch_status = st.empty()
        with col_stat3:
            time_status = st.empty()
        with col_stat4:
            acc_status = st.empty()
        
        # Live metrics
        st.markdown("---")
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.subheader("Accuracy Evolution")
            chart_accuracy = st.empty()
        
        with col_metric2:
            st.subheader("Forgetting Analysis")
            chart_forgetting = st.empty()
        
        # Buffer Analysis Section
        st.markdown("---")
        col_buffer1, col_buffer2 = st.columns(2)
        
        with col_buffer1:
            st.subheader("Buffer Distribution")
            chart_buffer_dist = st.empty()
        
        with col_buffer2:
            st.subheader("Validation Results (30% Test Set)")
            chart_per_class = st.empty()
        
        # Buffer Effectiveness Metrics
        st.markdown("---")
        st.subheader("Buffer Effectiveness Analysis")
        col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
        with col_eff1:
            buffer_util_metric = st.empty()
        with col_eff2:
            buffer_samples_metric = st.empty()
        with col_eff3:
            avg_forgetting_metric = st.empty()
        with col_eff4:
            retention_metric = st.empty()
        
        # Detailed table
        st.markdown("---")
        st.subheader("Detailed Metrics")
        table_metrics = st.empty()
        
        # Training data storage
        training_data = {
            'tasks': [],
            'classes': [],
            'accuracies': [],  # List of lists: each task's accuracy on all tasks so far
            'training_times': [],
            'forgetting': [],
            'buffer_stats': []  # Buffer statistics after each task
        }
        
        try:
            start_time = time.time()
            task_status.info("Initializing model...")
            
            # Initialize model based on phase (trainers will handle .to(device))
            if "Experience Replay" == phase:
                model = SimpleCNNMulticlass(num_classes=10)
                trainer = TrueContinualTrainer(
                    model=model,
                    use_replay=use_replay,
                    device=st.session_state.device,
                    num_tasks=num_tasks
                )
            elif "PEFT" in phase:
                model = SimpleCNNMulticlass(num_classes=10)
                trainer = PEFTContinualTrainer(
                    model=model,
                    use_replay=use_replay,
                    device=st.session_state.device,
                    num_tasks=num_tasks,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.0,
                    unfreeze_backbone=True,
                    num_classes=10  # Explicitly passing the number of classes
                )
            elif "Multi-Modal" in phase:
                vision_model = SimpleCNNMulticlass(num_classes=10)
                text_encoder = SimpleTextEncoder(
                    vocab_size=get_tokenizer().vocab_size,
                    embed_dim=256,
                    num_layers=2,
                    num_heads=4
                )
                multimodal_model = MultiModalClassifier(
                    vision_encoder=vision_model,
                    text_encoder=text_encoder,
                    num_classes=10,
                    fusion_type=fusion_type,
                    hidden_dim=256
                )
                trainer = MultiModalContinualTrainer(
                    multimodal_model=multimodal_model,
                    use_replay=use_replay,
                    device=st.session_state.device,
                    num_tasks=num_tasks,
                    text_mode=text_mode
                )
            else:  # Hardware Optimization
                model = SimpleCNNMulticlass(num_classes=10)
                trainer = HardwareContinualTrainer(
                    model=model,
                    device=st.session_state.device,
                    compression_strategy=compression_strategy,
                    target_hardware=target_hardware,
                    pruning_schedule=[0.0, 0.1, 0.2, 0.3, 0.5]
                )
            
            # Load dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            data_root = os.path.join(SCRIPT_DIR, "data")
            train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
            
            # Custom training callback for UI updates
            class UICallback:
                def __init__(self, ui_components, num_tasks, epochs_per_task, start_time):
                    self.ui = ui_components
                    self.num_tasks = num_tasks
                    self.epochs_per_task = epochs_per_task
                    self.start_time = start_time
                    self.current_task = 0
                    self.current_epoch = 0
                
                def on_task_start(self, task_id, task_classes):
                    self.current_task = task_id
                    class_names_str = f"{CLASS_NAMES[task_classes[0]]}, {CLASS_NAMES[task_classes[1]]}"
                    training_data['classes'].append(class_names_str)
                    self.ui['task_status'].info(f"Training Task {task_id+1}/{self.num_tasks}")
                    self.ui['task_info'].write(class_names_str)
                
                def on_epoch_start(self, task_id, epoch):
                    self.current_epoch = epoch
                    self.ui['epoch_status'].warning(f"Epoch {epoch+1}/{self.epochs_per_task}")
                    self.ui['epoch_info'].write(f"{epoch+1}/{self.epochs_per_task}")
                
                def on_batch_end(self, task_id, epoch, batch_idx, num_batches, loss):
                    # Update progress bars
                    overall_prog = (task_id * self.epochs_per_task + epoch + (batch_idx+1)/num_batches) / (self.num_tasks * self.epochs_per_task)
                    self.ui['overall_progress'].progress(min(overall_prog, 1.0))
                    self.ui['progress_pct'].write(f"{overall_prog*100:.1f}%")
                    
                    task_prog = (epoch + (batch_idx+1)/num_batches) / self.epochs_per_task
                    self.ui['task_progress'].progress(min(task_prog, 1.0))
                    
                    epoch_prog = (batch_idx + 1) / num_batches
                    self.ui['epoch_progress'].progress(epoch_prog)
                    
                    # Update time
                    elapsed = time.time() - self.start_time
                    self.ui['time_status'].success(f"Elapsed: {elapsed/60:.1f} min")
                
                def on_epoch_end(self, task_id, epoch, avg_loss):
                    self.ui['epoch_info'].write(f"{epoch+1}/{self.epochs_per_task} (Loss: {avg_loss:.4f})")
                
                def on_task_end(self, task_id, val_per_class, prev_tasks_val_results=None, buffer_stats=None):
                    """Display per-class validation results after task completion"""
                    # Store buffer statistics for UI visualization
                    if buffer_stats:
                        if 'buffer_stats' not in training_data:
                            training_data['buffer_stats'] = []
                        training_data['buffer_stats'].append(buffer_stats)
                    # Results displayed in terminal and charts below
            
            # Setup UI callback
            ui_components = {
                'task_status': task_status,
                'task_info': task_info,
                'epoch_status': epoch_status,
                'epoch_info': epoch_info,
                'overall_progress': overall_progress,
                'progress_pct': progress_pct,
                'task_progress': task_progress,
                'epoch_progress': epoch_progress,
                'time_status': time_status
            }
            
            callback = UICallback(ui_components, num_tasks, epochs_per_task, start_time)
            
            # Train using original trainer method with UI callbacks
            for task_id in range(num_tasks):
                task_start_time = time.time()
                task_classes = [task_id*2, task_id*2+1]
                
                callback.on_task_start(task_id, task_classes)
                
                # Get task data
                task_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
                task_subset = torch.utils.data.Subset(train_dataset, task_indices)
                
                # LoRA optimization: Use larger batch size for memory efficiency
                effective_batch_size = batch_size
                if "PEFT" in phase:
                    # LoRA uses 4% params → can fit 2x larger batches without OOM
                    effective_batch_size = min(batch_size, 512)  # Already optimized in UI, just ensure cap
                
                task_loader = DataLoader(task_subset, batch_size=effective_batch_size, shuffle=True)
                
                # Train based on phase type
                if "Multi-Modal" in phase:
                    # Multi-Modal uses trainer's own train_one_task method with callback
                    trainer.train_one_task(
                        train_loader=task_loader,
                        test_loader=None,  # We'll evaluate manually after
                        epochs=epochs_per_task,
                        lr=lr,
                        callback=callback,
                        task_id=task_id
                    )
                else:
                    # Phase 1, 2, 4 use standard train_one_task() with callback
                    train_one_task(
                        model=trainer.model,
                        train_loader=task_loader,
                        test_loader=None,
                        device=trainer.device,
                        epochs=epochs_per_task,
                        replay_buffer=trainer.replay_buffer if (use_replay and hasattr(trainer, 'replay_buffer')) else None,
                        callback=callback,
                        task_id=task_id
                    )
                
                # Evaluate after task
                task_time = time.time() - task_start_time
                training_data['training_times'].append(task_time)
                
                # Test on all tasks so far
                trainer.model.eval()
                task_accuracies = []
                
                with torch.no_grad():
                    for test_task_id in range(task_id + 1):
                        test_classes = [test_task_id*2, test_task_id*2+1]
                        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in test_classes]
                        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
                        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
                        
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(trainer.device)
                            labels = labels.to(trainer.device)
                            
                            # Handle Multi-Modal evaluation
                            if "Multi-Modal" in phase:
                                # Get text descriptions for labels
                                texts = [get_text_description(int(label), mode=text_mode) for label in labels]
                                input_ids, attention_mask = encode_texts(texts, device=trainer.device)
                                outputs = trainer.model(images, input_ids, attention_mask)
                            else:
                                outputs = trainer.model(images)
                            
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        
                        acc = correct / total
                        task_accuracies.append(acc)
                
                training_data['accuracies'].append(task_accuracies)
                
                # Get buffer statistics if available
                if use_replay and hasattr(trainer, 'replay_buffer') and trainer.replay_buffer:
                    buffer_stats = trainer.replay_buffer.get_statistics()
                    training_data['buffer_stats'].append(buffer_stats)
                else:
                    training_data['buffer_stats'].append(None)
                
                # Notify callback with buffer stats
                if callback:
                    callback.on_task_end(
                        task_id=task_id,
                        val_per_class=None,
                        prev_tasks_val_results=None,
                        buffer_stats=training_data['buffer_stats'][-1]
                    )
                
                # Calculate forgetting
                if task_id > 0:
                    forgetting_vals = []
                    for prev_task in range(task_id):
                        initial = training_data['accuracies'][prev_task][prev_task]
                        current = task_accuracies[prev_task]
                        forgetting_vals.append((initial - current) * 100)
                    avg_forgetting = np.mean(forgetting_vals)
                    training_data['forgetting'].append(avg_forgetting)
                else:
                    training_data['forgetting'].append(0.0)
                
                # Update accuracy status
                avg_acc = np.mean(task_accuracies)
                acc_status.metric("Avg Acc", f"{avg_acc*100:.1f}%")
                
                # Update accuracy chart
                fig_acc = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for i in range(len(training_data['accuracies'])):
                    x_vals = list(range(i, len(training_data['accuracies'])))
                    y_vals = [training_data['accuracies'][j][i] * 100 for j in x_vals]
                    
                    fig_acc.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=f'Task {i}',
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=8)
                    ))
                
                fig_acc.update_layout(
                    title="Accuracy Evolution",
                    xaxis_title="After Training Task",
                    yaxis_title="Accuracy (%)",
                    xaxis=dict(tickmode='linear', dtick=1),
                    yaxis=dict(range=[0, 105], ticksuffix='%'),
                    hovermode='x unified',
                    height=400,
                    margin=dict(l=60, r=40, t=60, b=60),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                chart_accuracy.plotly_chart(fig_acc, use_container_width=True)
                
                # Update forgetting chart
                if len(training_data['forgetting']) > 1:
                    fig_forg = go.Figure()
                    fig_forg.add_trace(go.Bar(
                        x=[f"Task {i}" for i in range(len(training_data['forgetting']))],
                        y=training_data['forgetting'],
                        marker_color=['lightgray'] + ['#FF4B4B'] * (len(training_data['forgetting'])-1),
                        text=[f"{f:+.2f}%" for f in training_data['forgetting']],
                        textposition='inside',
                        textfont=dict(size=12, color='white')
                    ))
                    
                    # Calculate y-axis range to accommodate all values
                    max_forg = max(training_data['forgetting']) if training_data['forgetting'] else 0
                    min_forg = min(training_data['forgetting']) if training_data['forgetting'] else 0
                    y_max = max(max_forg * 1.2, 10)
                    y_min = min(min_forg * 1.2, -10) if min_forg < 0 else 0
                    
                    fig_forg.update_layout(
                        title="Catastrophic Forgetting Analysis",
                        yaxis_title="Forgetting (%)",
                        yaxis=dict(range=[y_min, y_max], ticksuffix='%'),
                        height=400,
                        margin=dict(l=60, r=40, t=60, b=60),
                        showlegend=False
                    )
                    
                    chart_forgetting.plotly_chart(fig_forg, use_container_width=True)
                
                # Update Buffer Distribution Chart
                if use_replay and training_data['buffer_stats'][-1] is not None:
                    buffer_stats = training_data['buffer_stats'][-1]
                    
                    # Extract per-class counts
                    per_class_count = buffer_stats['per_class_count']
                    
                    classes_in_buffer = sorted(per_class_count.keys())
                    counts = [per_class_count[c] for c in classes_in_buffer]
                    class_labels = [f"Class {c}\n{CLASS_NAMES[c]}" for c in classes_in_buffer]
                    
                    # Create simple bar chart showing actual data
                    fig_buffer = go.Figure()
                    
                    # Add bars for actual samples
                    fig_buffer.add_trace(go.Bar(
                        x=class_labels,
                        y=counts,
                        marker_color='#4CAF50',
                        text=[f"{count}" for count in counts],
                        textposition='inside',
                        textfont=dict(size=12, color='white'),
                        hovertemplate='%{x}<br>Samples: %{y}<extra></extra>'
                    ))
                    
                    fig_buffer.update_layout(
                        title=f"Buffer Distribution (Task {task_id}) - Total: {buffer_stats['total_samples']}/5000 samples",
                        xaxis_title="Class",
                        yaxis_title="Samples per Class",
                        height=400,
                        showlegend=False
                    )
                    
                    chart_buffer_dist.plotly_chart(fig_buffer, use_container_width=True)
                    
                    # Update buffer effectiveness metrics
                    utilization = buffer_stats['utilization']
                    total_samples = buffer_stats['total_samples']
                    
                    buffer_util_metric.metric(
                        "Buffer Utilization",
                        f"{utilization:.1f}%",
                        delta=None
                    )
                    
                    buffer_samples_metric.metric(
                        "Total Samples",
                        f"{total_samples}/5000",
                        delta=None
                    )
                    
                    if task_id > 0:
                        avg_forg_val = training_data['forgetting'][-1]
                        retention_val = 100 - avg_forg_val
                        
                        avg_forgetting_metric.metric(
                            "Avg Forgetting",
                            f"{avg_forg_val:.2f}%",
                            delta=f"{avg_forg_val - training_data['forgetting'][-2]:.2f}%" if task_id > 1 else None,
                            delta_color="inverse"
                        )
                        
                        retention_metric.metric(
                            "Knowledge Retained",
                            f"{retention_val:.1f}%",
                            delta=None
                        )
                
                # Update Validation Results Table
                validation_table_data = []
                for test_task_id in range(task_id + 1):
                    task_acc = training_data['accuracies'][task_id][test_task_id]
                    
                    # Get 30% validation data for this task
                    test_classes = [test_task_id*2, test_task_id*2+1]
                    from data.fashion_mnist_true_continual import get_task_loaders_true_continual
                    _, val_loader_temp, _, _ = get_task_loaders_true_continual(
                        test_task_id, batch_size=128, root=data_root, train_ratio=0.7
                    )
                    
                    # Get per-class accuracy on validation set
                    class_correct = {c: 0 for c in test_classes}
                    class_total = {c: 0 for c in test_classes}
                    
                    trainer.model.eval()
                    with torch.no_grad():
                        for images, labels in val_loader_temp:
                            images = images.to(trainer.device)
                            labels = labels.to(trainer.device)
                            
                            if "Multi-Modal" in phase:
                                texts = [get_text_description(int(label), mode=text_mode) for label in labels]
                                input_ids, attention_mask = encode_texts(texts, device=trainer.device)
                                outputs = trainer.model(images, input_ids, attention_mask)
                            else:
                                outputs = trainer.model(images)
                            
                            _, predicted = torch.max(outputs, 1)
                            
                            for pred, label in zip(predicted, labels):
                                label_item = int(label)
                                if label_item in class_total:
                                    class_total[label_item] += 1
                                    if pred == label:
                                        class_correct[label_item] += 1
                    
                    # Build table row
                    for c in test_classes:
                        if class_total[c] > 0:
                            acc = (class_correct[c] / class_total[c]) * 100
                            validation_table_data.append({
                                'Task': f"Task {test_task_id}",
                                'Class': f"C{c} {CLASS_NAMES[c]}",
                                'Correct': class_correct[c],
                                'Total': class_total[c],
                                'Accuracy': f"{acc:.2f}%"
                            })
                
                # Display validation results table
                if validation_table_data:
                    df_val = pd.DataFrame(validation_table_data)
                    chart_per_class.dataframe(df_val, use_container_width=True, hide_index=True)
                
                # Update table
                table_data = []
                for i in range(len(training_data['accuracies'])):
                    row = {
                        'Task': f"{i}",
                        'Classes': training_data['classes'][i],
                        'Time (min)': f"{training_data['training_times'][i]/60:.2f}",
                        'Task Acc': f"{training_data['accuracies'][i][i]*100:.2f}%",
                        'Avg Acc': f"{np.mean(training_data['accuracies'][i])*100:.2f}%",
                        'Forgetting': f"{training_data['forgetting'][i]:+.2f}%" if i > 0 else "—"
                    }
                    table_data.append(row)
                
                df = pd.DataFrame(table_data)
                
                # Style the dataframe
                def highlight_best(s):
                    if s.name == 'Avg Acc':
                        max_val = df['Avg Acc'].max()
                        return ['background-color: #90EE90' if v == max_val else '' for v in s]
                    return ['' for _ in s]
                
                styled_df = df.style.apply(highlight_best)
                table_metrics.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Save checkpoint after each task
                checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                phase_names = {
                    "Experience Replay": "phase1",
                    "PEFT/LoRA": "phase2",
                    "Multi-Modal (Vision + Text)": "phase3",
                    "Hardware Optimization": "phase4"
                }
                phase_name = phase_names.get(phase, "model")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"{phase_name}_task{task_id}.pt")
                torch.save({
                    'model_state_dict': trainer.model.state_dict(),
                    'task_id': task_id,
                    'metrics': {
                        'task_accuracy': training_data['accuracies'][task_id][task_id],
                        'avg_accuracy': np.mean(training_data['accuracies'][task_id]),
                        'forgetting': training_data['forgetting'][task_id],
                        'training_time': training_data['training_times'][task_id]
                    },
                    'phase': phase,
                    'config': {
                        'num_tasks': task_id + 1,
                        'epochs_per_task': epochs_per_task,
                        'batch_size': batch_size,
                        'lr': lr
                    },
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, checkpoint_path)
                
                # Get checkpoint file size
                checkpoint_size_bytes = os.path.getsize(checkpoint_path)
                checkpoint_size_mb = checkpoint_size_bytes / (1024 * 1024)
            
            # Training complete
            overall_progress.progress(1.0)
            progress_pct.write("100%")
            task_status.success("Training Complete")
            
            # Apply hardware optimization compression for Phase 4
            if phase == "Hardware Optimization":
                st.markdown("---")
                st.subheader("🔧 Hardware Optimization - Compression")
                
                compress_status = st.empty()
                compress_status.info("Applying model compression (quantization + pruning)...")
                
                try:
                    # Get model size before compression
                    pre_compress_checkpoint = os.path.join(checkpoint_dir, f"{phase_name}_precompress.pt")
                    torch.save(trainer.model.state_dict(), pre_compress_checkpoint)
                    pre_compress_size_bytes = os.path.getsize(pre_compress_checkpoint)
                    pre_compress_size_mb = pre_compress_size_bytes / (1024 * 1024)
                    
                    # Apply compression
                    compression_metrics = trainer.compress_final()
                    
                    # Get model size after compression
                    post_compress_checkpoint = os.path.join(checkpoint_dir, f"{phase_name}_compressed.pt")
                    torch.save(trainer.model.state_dict(), post_compress_checkpoint)
                    post_compress_size_bytes = os.path.getsize(post_compress_checkpoint)
                    post_compress_size_mb = post_compress_size_bytes / (1024 * 1024)
                    
                    compression_ratio = pre_compress_size_mb / post_compress_size_mb
                    
                    compress_status.success(f"✅ Compression Complete: {pre_compress_size_mb:.2f} MB → {post_compress_size_mb:.2f} MB ({compression_ratio:.2f}x reduction)")
                    
                    # Update checkpoint_size_mb for summary display
                    checkpoint_size_mb = post_compress_size_mb
                    
                    # Show compression details
                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                    with col_comp1:
                        st.metric("Before Compression", f"{pre_compress_size_mb:.2f} MB")
                    with col_comp2:
                        st.metric("After Compression", f"{post_compress_size_mb:.2f} MB", delta=f"-{pre_compress_size_mb - post_compress_size_mb:.2f} MB")
                    with col_comp3:
                        st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
                    
                    # Cleanup temporary checkpoints
                    os.remove(pre_compress_checkpoint)
                    
                except Exception as e:
                    compress_status.error(f"Compression failed: {str(e)}")
                    st.warning("Continuing with uncompressed model...")
            
            # Save trainer
            st.session_state.trainer = trainer
            st.session_state.training_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'phase': phase,
                'data': training_data,
                'total_time': time.time() - start_time
            })
            
            # Summary
            st.markdown("---")
            st.subheader("Training Summary")
            
            sum1, sum2, sum3, sum4, sum5 = st.columns(5)
            with sum1:
                final_avg = np.mean(training_data['accuracies'][-1])
                st.metric("Final Avg Accuracy", f"{final_avg*100:.2f}%")
            with sum2:
                total_time = time.time() - start_time
                st.metric("Total Time", f"{total_time/60:.2f} min")
            with sum3:
                avg_time = np.mean(training_data['training_times'])
                st.metric("Avg Time/Task", f"{avg_time/60:.2f} min")
            with sum4:
                if len(training_data['forgetting']) > 1:
                    avg_forg = np.mean(training_data['forgetting'][1:])
                    st.metric("Avg Forgetting", f"{avg_forg:.2f}%")
            with sum5:
                # Use last checkpoint size as reference
                if 'checkpoint_size_mb' in locals():
                    st.metric("Model Size", f"{checkpoint_size_mb:.2f} MB")
            
            # Save final checkpoint with all training data
            checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            phase_names = {
                "Experience Replay": "phase1",
                "PEFT/LoRA": "phase2",
                "Multi-Modal (Vision + Text)": "phase3",
                "Hardware Optimization": "phase4"
            }
            phase_name = phase_names.get(phase, "model")
            
            final_checkpoint_path = os.path.join(checkpoint_dir, f"{phase_name}_final.pt")
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'metrics': training_data,
                'training_time': total_time,
                'phase': phase,
                'config': {
                    'num_tasks': num_tasks,
                    'epochs_per_task': epochs_per_task,
                    'batch_size': batch_size,
                    'lr': lr
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, final_checkpoint_path)
            
            # Get final checkpoint size
            final_model_size_bytes = os.path.getsize(final_checkpoint_path)
            final_model_size_mb = final_model_size_bytes / (1024 * 1024)
            
            st.success(f"✅ {num_tasks} task checkpoints saved to: checkpoints/{phase_name}_task*.pt")
            st.success(f"✅ Final checkpoint saved: {final_checkpoint_path}")
            st.info(f"📊 Model Size: {final_model_size_mb:.2f} MB ({final_model_size_bytes:,} bytes)")
            
            # Export report
            st.markdown("---")
            export_text = f"""CONTINUAL LEARNING TRAINING REPORT
{'='*70}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Strategy: {phase}
Device: {st.session_state.device.upper()}

CONFIGURATION:
- Number of Tasks: {num_tasks}
- Epochs per Task: {epochs_per_task}
- Batch Size: {batch_size}
- Learning Rate: {lr}
- Final Model Size: {final_model_size_mb:.2f} MB ({final_model_size_bytes:,} bytes)

RESULTS:
{'='*70}
"""
            for i, row in enumerate(table_data):
                export_text += f"\nTask {i}: {row['Classes']}\n"
                export_text += f"  Training Time: {row['Time (min)']} minutes\n"
                export_text += f"  Task Accuracy: {row['Task Acc']}\n"
                export_text += f"  Average Accuracy: {row['Avg Acc']}\n"
                export_text += f"  Forgetting: {row['Forgetting']}\n"
            
            export_text += f"\n{'='*70}\n"
            export_text += f"SUMMARY:\n"
            export_text += f"- Final Average Accuracy: {final_avg*100:.2f}%\n"
            export_text += f"- Total Training Time: {total_time/60:.2f} minutes\n"
            export_text += f"- Average Time per Task: {avg_time/60:.2f} minutes\n"
            if len(training_data['forgetting']) > 1:
                export_text += f"- Average Forgetting: {avg_forg:.2f}%\n"
            export_text += f"- Final Model Size: {final_model_size_mb:.2f} MB ({final_model_size_bytes:,} bytes)\n"
            export_text += f"\nCHECKPOINTS:\n"
            export_text += f"- Task Checkpoints: checkpoints/{phase_name}_task0.pt to task{num_tasks-1}.pt\n"
            export_text += f"- Final Checkpoint: {final_checkpoint_path}\n"
            export_text += f"{'='*70}\n"
            
            st.download_button(
                label="Download Training Report",
                data=export_text,
                file_name=f"training_report_{phase_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        except Exception as e:
            task_status.error(f"Error: {str(e)}")
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ===== TESTING TAB =====
with tab2:
    st.header("Model Testing")
    
    if st.session_state.trainer is None:
        st.warning("No trained model available. Please train a model first.")
    else:
        st.success("Model loaded and ready for testing")
        
        # Test mode selection
        test_mode = st.radio("Test Mode", ["Upload Image", "Random from Dataset", "Batch Evaluation"], horizontal=True)
        
        # ===== UPLOAD IMAGE MODE =====
        if test_mode == "Upload Image":
            st.subheader("Upload an Image")
            uploaded_file = st.file_uploader("Choose an image (28x28 grayscale recommended)", type=['png', 'jpg', 'jpeg'])
            
            col1, col2 = st.columns(2)
            
            if uploaded_file is not None:
                # Load and preprocess image
                image = Image.open(uploaded_file).convert('L')
                
                with col1:
                    st.image(image, caption="Uploaded Image", width=200)
                
                # Preprocessing options
                with st.expander("Image Preprocessing Options"):
                    resize_method = st.selectbox("Resize Method", ["Fit to 28x28", "Crop Center", "Pad to Square"])
                    contrast = st.slider("Contrast Enhancement", 0.5, 2.0, 1.0, 0.1)
                    brightness = st.slider("Brightness Adjustment", 0.5, 2.0, 1.0, 0.1)
                    apply_blur = st.checkbox("Apply Slight Blur", value=False)
                    invert = st.checkbox("Invert Colors", value=False)
                
                # Preprocess
                if resize_method == "Fit to 28x28":
                    image = image.resize((28, 28), Image.Resampling.LANCZOS)
                elif resize_method == "Crop Center":
                    width, height = image.size
                    size = min(width, height)
                    left = (width - size) // 2
                    top = (height - size) // 2
                    image = image.crop((left, top, left + size, top + size))
                    image = image.resize((28, 28), Image.Resampling.LANCZOS)
                else:  # Pad to Square
                    width, height = image.size
                    size = max(width, height)
                    new_image = Image.new('L', (size, size), 0)
                    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
                    image = new_image.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Apply enhancements
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(contrast)
                
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(brightness)
                
                if apply_blur:
                    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                if invert:
                    image = ImageOps.invert(image)
                
                with col2:
                    st.image(image, caption="Preprocessed Image (28x28)", width=200)
                
                # Convert to tensor
                img_array = np.array(image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
                img_tensor = (img_tensor - 0.5) / 0.5  # Normalize
                img_tensor = img_tensor.to(st.session_state.device)
                
                # Predict
                if st.button("Classify Image", type="primary", use_container_width=True):
                    with st.spinner("Classifying..."):
                        st.session_state.trainer.model.eval()
                        with torch.no_grad():
                            # Handle Multi-Modal models
                            if "Multi-Modal" in phase:
                                # Create placeholder text for uploaded images
                                text = "uploaded fashion item"
                                input_ids, attention_mask = encode_texts([text], device=st.session_state.device)
                                outputs = st.session_state.trainer.model(img_tensor, input_ids, attention_mask)
                            else:
                                outputs = st.session_state.trainer.model(img_tensor)
                            
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            predicted_class = predicted.item()
                            confidence_pct = confidence.item() * 100
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.metric("Predicted Class", CLASS_NAMES[predicted_class])
                            st.metric("Confidence", f"{confidence_pct:.2f}%")
                        
                        with result_col2:
                            # Show top 3 predictions
                            top3_prob, top3_idx = torch.topk(probabilities[0], 3)
                            st.write("**Top 3 Predictions:**")
                            for i in range(3):
                                st.write(f"{i+1}. {CLASS_NAMES[top3_idx[i].item()]}: {top3_prob[i].item()*100:.2f}%")
                        
                        # Confidence bar chart
                        all_probs = probabilities[0].cpu().numpy()
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[CLASS_NAMES[i] for i in range(10)],
                                y=all_probs * 100,
                                marker_color=['#FF4B4B' if i == predicted_class else '#CCCCCC' for i in range(10)]
                            )
                        ])
                        fig.update_layout(
                            title="Class Probabilities",
                            xaxis_title="Class",
                            yaxis_title="Probability (%)",
                            yaxis=dict(range=[0, 100]),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # ===== RANDOM FROM DATASET MODE =====
        elif test_mode == "Random from Dataset":
            st.subheader("Test on Random Samples")
            
            # Load test dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            data_root = os.path.join(SCRIPT_DIR, "data")
            test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                num_samples = st.slider("Number of Samples", 1, 10, 5)
                filter_class = st.selectbox("Filter by Class", ["All"] + CLASS_NAMES)
            
            if st.button("Generate Random Samples", type="primary", use_container_width=True):
                # Get random samples
                if filter_class == "All":
                    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
                else:
                    class_idx = CLASS_NAMES.index(filter_class)
                    class_indices = [i for i, (_, label) in enumerate(test_dataset) if label == class_idx]
                    indices = np.random.choice(class_indices, min(num_samples, len(class_indices)), replace=False)
                
                st.markdown("---")
                st.subheader("Predictions")
                
                # Create grid
                cols_per_row = 5
                for i in range(0, len(indices), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, idx in enumerate(indices[i:i+cols_per_row]):
                        with cols[j]:
                            img, true_label = test_dataset[idx]
                            
                            # Display image
                            img_display = img.squeeze().numpy()
                            img_display = (img_display * 0.5 + 0.5)  # Denormalize
                            st.image(img_display, width=100, clamp=True)
                            
                            # Predict
                            st.session_state.trainer.model.eval()
                            with torch.no_grad():
                                img_input = img.unsqueeze(0).to(st.session_state.device)
                                
                                # Handle Multi-Modal models
                                if "Multi-Modal" in phase:
                                    text = get_text_description(true_label, mode=text_mode if 'text_mode' in dir() else 'simple')
                                    input_ids, attention_mask = encode_texts([text], device=st.session_state.device)
                                    outputs = st.session_state.trainer.model(img_input, input_ids, attention_mask)
                                else:
                                    outputs = st.session_state.trainer.model(img_input)
                                
                                probabilities = torch.softmax(outputs, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                predicted_class = predicted.item()
                                confidence_pct = confidence.item() * 100
                            
                            # Display results
                            true_name = CLASS_NAMES[true_label]
                            pred_name = CLASS_NAMES[predicted_class]
                            
                            if predicted_class == true_label:
                                st.success(f"✓ {pred_name}")
                                st.caption(f"{confidence_pct:.1f}%")
                            else:
                                st.error(f"✗ {pred_name}")
                                st.caption(f"True: {true_name}")
                                st.caption(f"{confidence_pct:.1f}%")
        
        # ===== BATCH EVALUATION MODE =====
        else:  # Batch Evaluation
            st.subheader("Batch Evaluation")
            
            # Load test dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            data_root = os.path.join(SCRIPT_DIR, "data")
            test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
            
            eval_option = st.radio("Evaluation Scope", ["All Tasks", "Specific Task", "Specific Class"], horizontal=True)
            
            if eval_option == "All Tasks":
                num_tasks_trained = len(st.session_state.trainer.replay_buffer.data) if hasattr(st.session_state.trainer, 'replay_buffer') and st.session_state.trainer.replay_buffer else num_tasks
                
                if st.button("Evaluate All Tasks", type="primary", use_container_width=True):
                    with st.spinner("Evaluating..."):
                        st.session_state.trainer.model.eval()
                        
                        results = []
                        confusion_matrix = np.zeros((10, 10))
                        
                        with torch.no_grad():
                            for task_id in range(num_tasks_trained):
                                task_classes = [task_id*2, task_id*2+1]
                                task_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]
                                task_subset = torch.utils.data.Subset(test_dataset, task_indices)
                                task_loader = DataLoader(task_subset, batch_size=128, shuffle=False)
                                
                                correct = 0
                                total = 0
                                
                                for images, labels in task_loader:
                                    images = images.to(st.session_state.device)
                                    labels = labels.to(st.session_state.device)
                                    
                                    # Handle Multi-Modal models
                                    if "Multi-Modal" in phase:
                                        texts = [get_text_description(int(label), mode=text_mode if 'text_mode' in dir() else 'simple') for label in labels]
                                        input_ids, attention_mask = encode_texts(texts, device=st.session_state.device)
                                        outputs = st.session_state.trainer.model(images, input_ids, attention_mask)
                                    else:
                                        outputs = st.session_state.trainer.model(images)
                                    
                                    _, predicted = torch.max(outputs, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                                    
                                    # Update confusion matrix
                                    for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                                        confusion_matrix[t][p] += 1
                                
                                accuracy = correct / total
                                results.append({
                                    'Task': f"Task {task_id}",
                                    'Classes': f"{CLASS_NAMES[task_classes[0]]}, {CLASS_NAMES[task_classes[1]]}",
                                    'Accuracy': f"{accuracy*100:.2f}%",
                                    'Correct': correct,
                                    'Total': total
                                })
                        
                        # Display results table
                        st.markdown("---")
                        st.subheader("Task-wise Results")
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Overall accuracy
                        overall_acc = sum([r['Correct'] for r in results]) / sum([r['Total'] for r in results])
                        st.metric("Overall Accuracy", f"{overall_acc*100:.2f}%")
                        
                        # Confusion Matrix Heatmap
                        st.markdown("---")
                        st.subheader("Confusion Matrix")
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=confusion_matrix,
                            x=CLASS_NAMES,
                            y=CLASS_NAMES,
                            colorscale='Blues',
                            text=confusion_matrix.astype(int),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Confusion Matrix",
                            xaxis_title="Predicted Class",
                            yaxis_title="True Class",
                            height=600,
                            xaxis={'side': 'bottom'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif eval_option == "Specific Task":
                task_id = st.selectbox("Select Task", range(num_tasks))
                
                if st.button("Evaluate Task", type="primary", use_container_width=True):
                    with st.spinner("Evaluating..."):
                        task_classes = [task_id*2, task_id*2+1]
                        task_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]
                        task_subset = torch.utils.data.Subset(test_dataset, task_indices)
                        task_loader = DataLoader(task_subset, batch_size=128, shuffle=False)
                        
                        st.session_state.trainer.model.eval()
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for images, labels in task_loader:
                                images = images.to(st.session_state.device)
                                labels = labels.to(st.session_state.device)
                                
                                # Handle Multi-Modal models
                                if "Multi-Modal" in phase:
                                    texts = [get_text_description(int(label), mode=text_mode if 'text_mode' in dir() else 'simple') for label in labels]
                                    input_ids, attention_mask = encode_texts(texts, device=st.session_state.device)
                                    outputs = st.session_state.trainer.model(images, input_ids, attention_mask)
                                else:
                                    outputs = st.session_state.trainer.model(images)
                                
                                _, predicted = torch.max(outputs, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                        
                        accuracy = correct / total
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Task", f"Task {task_id}")
                        with col2:
                            st.metric("Classes", f"{CLASS_NAMES[task_classes[0]]}, {CLASS_NAMES[task_classes[1]]}")
                        with col3:
                            st.metric("Accuracy", f"{accuracy*100:.2f}%")
            
            else:  # Specific Class
                class_name = st.selectbox("Select Class", CLASS_NAMES)
                class_idx = CLASS_NAMES.index(class_name)
                
                if st.button("Evaluate Class", type="primary", use_container_width=True):
                    with st.spinner("Evaluating..."):
                        class_indices = [i for i, (_, label) in enumerate(test_dataset) if label == class_idx]
                        class_subset = torch.utils.data.Subset(test_dataset, class_indices)
                        class_loader = DataLoader(class_subset, batch_size=128, shuffle=False)
                        
                        st.session_state.trainer.model.eval()
                        correct = 0
                        total = 0
                        predictions_count = {name: 0 for name in CLASS_NAMES}
                        
                        with torch.no_grad():
                            for images, labels in class_loader:
                                images = images.to(st.session_state.device)
                                labels = labels.to(st.session_state.device)
                                
                                # Handle Multi-Modal models
                                if "Multi-Modal" in phase:
                                    texts = [get_text_description(int(label), mode=text_mode if 'text_mode' in dir() else 'simple') for label in labels]
                                    input_ids, attention_mask = encode_texts(texts, device=st.session_state.device)
                                    outputs = st.session_state.trainer.model(images, input_ids, attention_mask)
                                else:
                                    outputs = st.session_state.trainer.model(images)
                                
                                _, predicted = torch.max(outputs, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                                
                                # Count predictions
                                for p in predicted.cpu().numpy():
                                    predictions_count[CLASS_NAMES[p]] += 1
                        
                        accuracy = correct / total
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Class", class_name)
                        with col2:
                            st.metric("Accuracy", f"{accuracy*100:.2f}%")
                        with col3:
                            st.metric("Samples", total)
                        
                        # Prediction distribution
                        st.markdown("---")
                        st.subheader("Prediction Distribution")
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(predictions_count.keys()),
                                y=list(predictions_count.values()),
                                marker_color=['#FF4B4B' if name == class_name else '#CCCCCC' for name in predictions_count.keys()]
                            )
                        ])
                        fig.update_layout(
                            title=f"Where {class_name} samples were predicted",
                            xaxis_title="Predicted Class",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ===== HISTORY TAB =====
with tab3:
    st.header("Training History")
    
    if len(st.session_state.training_history) == 0:
        st.info("No training history yet. Train a model first!")
    else:
        for idx, history in enumerate(reversed(st.session_state.training_history)):
            with st.expander(f"{history['timestamp']} - {history['phase']}", expanded=(idx==0)):
                data = history['data']
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    final_avg = np.mean(data['accuracies'][-1])
                    st.metric("Final Avg Accuracy", f"{final_avg*100:.2f}%")
                with col2:
                    st.metric("Total Time", f"{history['total_time']/60:.2f} min")
                with col3:
                    st.metric("Tasks Completed", len(data['accuracies']))
                
                # Accuracy chart
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for i in range(len(data['accuracies'])):
                    x_vals = list(range(i, len(data['accuracies'])))
                    y_vals = [data['accuracies'][j][i] * 100 for j in x_vals]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=f'Task {i}',
                        line=dict(width=2, color=colors[i % len(colors)])
                    ))
                
                fig.update_layout(
                    title="Accuracy Evolution",
                    xaxis_title="After Training Task",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[0, 100]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Continual Learning System** | Fashion-MNIST Dataset | Strategies: Experience Replay, PEFT/LoRA, Multi-Modal, Hardware Optimization")

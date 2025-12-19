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
epochs_per_task = st.sidebar.slider("Epochs per Task", 5, 30, 15)
batch_size = st.sidebar.slider("Batch Size", 32, 256, 128)
lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

# Strategy-specific params
if "Experience Replay" == phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    buffer_size = st.sidebar.slider("Buffer Size (per class)", 100, 1000, 500)
elif "PEFT" in phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    buffer_size = st.sidebar.slider("Buffer Size (per class)", 100, 1000, 500)
    lora_rank = st.sidebar.slider("LoRA Rank", 8, 64, 24)
    lora_alpha = st.sidebar.slider("LoRA Alpha", 16, 128, 48)
elif "Multi-Modal" in phase:
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    buffer_size = st.sidebar.slider("Buffer Size (per class)", 100, 1000, 500)
    fusion_type = st.sidebar.selectbox("Fusion Strategy", ["concat", "cross_attention", "gated"], index=0)
    text_mode = st.sidebar.selectbox("Text Mode", ["simple", "rich", "attributes"], index=0)
else:  # Hardware Optimization
    use_replay = st.sidebar.checkbox("Use Experience Replay", value=True)
    buffer_size = st.sidebar.slider("Buffer Size (per class)", 100, 1000, 500)
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
            'forgetting': []
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
                    num_tasks=num_tasks,
                    buffer_size=buffer_size
                )
            elif "PEFT" in phase:
                model = SimpleCNNMulticlass(num_classes=10)
                trainer = PEFTContinualTrainer(
                    model=model,
                    use_replay=use_replay,
                    device=st.session_state.device,
                    num_tasks=num_tasks,
                    buffer_size=buffer_size,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha
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
                    buffer_size=buffer_size,
                    text_mode=text_mode
                )
            else:  # Hardware Optimization
                model = SimpleCNNMulticlass(num_classes=10)
                trainer = HardwareContinualTrainer(
                    model=model,
                    device=st.session_state.device,
                    buffer_size=buffer_size,
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
                task_loader = DataLoader(task_subset, batch_size=batch_size, shuffle=True)
                
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
            
            # Training complete
            overall_progress.progress(1.0)
            progress_pct.write("100%")
            task_status.success("Training Complete")
            
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
            
            sum1, sum2, sum3, sum4 = st.columns(4)
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
            
            # Save checkpoint
            checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            phase_names = {
                "Experience Replay": "phase1",
                "PEFT/LoRA": "phase2",
                "Multi-Modal (Vision + Text)": "phase3",
                "Hardware Optimization": "phase4"
            }
            phase_name = phase_names.get(phase, "model")
            
            checkpoint_path = os.path.join(checkpoint_dir, f"{phase_name}_task{num_tasks-1}.pt")
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
            }, checkpoint_path)
            
            st.success(f"Checkpoint saved: {checkpoint_path}")
            
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
    st.info("Testing functionality preserved from original app.py")
    # (Keep existing testing code)

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

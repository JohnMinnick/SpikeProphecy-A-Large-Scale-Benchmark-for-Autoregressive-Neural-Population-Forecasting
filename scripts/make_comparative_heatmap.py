import os
import sys
import boto3
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.models.common import create_teacher_model
from src.models.student import StudentSNN
from src.models.gac_snn import GacStudentSNN
from src.data.multi_session_loader import load_multi_session_nwb

import base64

def download_checkpoint(slug):
    local_path = f'{slug}_best_model.pt'
    if not os.path.exists(local_path):
        print(f"Downloading {slug} from S3...")
        s3 = boto3.client(
            's3',
            endpoint_url='https://s3-west.nrp-nautilus.io',
            aws_access_key_id=base64.b64decode("NEI2RUY0U0dOU0ExRUsyQ0dTTjg=").decode('utf-8'),
            aws_secret_access_key=base64.b64decode("RFQ2T3Q4WXJtUmxRbmFjWmVCU1c3d3ZWMVJLdXRBUGpSTE1uRW9PVA==").decode('utf-8')
        )
        bucket = 'braingeneersdev'
        key = f'jrm/spike-prophecy/outputs/{slug}/best_model.pt'
        s3.download_file(bucket, key, local_path)
    return local_path

@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Download Checkpoints
    snn_ckpt_path = download_checkpoint('snn-standalone-v12b')
    gac_ckpt_path = download_checkpoint('2026-03-26_gac-distill-v12')

    # 2. Get Data (Using one file)
    print("Loading data...")
    config = load_config("configs/student/distill_gac_snn.yaml")
    config['data'] = {
        'source': {
            'type': 'nwb_multi',
            'file_list': ['data/raw/Steinmetz2019_Cori_2016-12-14.nwb']
        },
        'bin_width_ms': 50,
        'history_bins': 50,
    }
    
    spike_counts, mask_index, metadata = load_multi_session_nwb(config['data'])
    m_max = metadata['m_max']
    
    # Extract validation window
    t_total = spike_counts.shape[1]
    train_end = int(t_total * 0.7)
    val_end = train_end + int(t_total * 0.15)
    
    # Let's take 200 bins from the start of val
    history_bins = 50
    start_t = train_end + history_bins
    window_len = 200
    
    x_sequence = spike_counts[:, start_t - history_bins : start_t + window_len - 1].T
    x_tensor_full = torch.tensor(x_sequence, dtype=torch.float32).unsqueeze(0).to(device) # (1, T+H-1, m_i)
    
    # Pad to global M_max=1240
    global_m_max = 1240
    if m_max < global_m_max:
        padded = torch.zeros(1, x_tensor_full.shape[1], global_m_max, device=device)
        padded[:, :, :m_max] = x_tensor_full
        x_tensor_full = padded
    
    # Format inputs for sliding window
    # Actually, the easiest way to run inference is to just pass the padded sequence and slice out the predictions.
    # The models are autoregressive / stateful.
    
    # Create SNN Standalone
    print("Loading SNN Standalone...")
    snn_cfg = load_config("configs/student/distill_nrp.yaml")
    snn_model_cfg = snn_cfg['model'].copy()
    if 'type' in snn_model_cfg: snn_model_cfg.pop('type')
    if 'input_size' in snn_model_cfg: snn_model_cfg.pop('input_size')
    
    snn = StudentSNN(input_size=global_m_max, **snn_model_cfg).to(device)
    state = torch.load(snn_ckpt_path, map_location=device)
    snn.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    snn.eval()
    
    # Create GAC
    print("Loading GAC Distill...")
    gac_cfg = load_config("configs/student/distill_gac_snn.yaml")
    gac_model_cfg = gac_cfg['model'].copy()
    if 'type' in gac_model_cfg: gac_model_cfg.pop('type')
    if 'input_size' in gac_model_cfg: gac_model_cfg.pop('input_size')
    if 'beta' in gac_model_cfg: gac_model_cfg['beta_init'] = gac_model_cfg.pop('beta')
    
    gac_snn = GacStudentSNN(input_size=global_m_max, **gac_model_cfg).to(device)
    state = torch.load(gac_ckpt_path, map_location=device)
    gac_snn.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    gac_snn.eval()
    
    print("Running inference...")
    
    # We need to construct sliding windows of length `history_bins`.
    # x_tensor_full shape is (1, 249, 1240).
    # We want to extract 200 windows.
    windows = []
    for t in range(window_len):
        windows.append(x_tensor_full[0, t:t+history_bins, :])
    
    # Stack to (200, 50, 1240)
    batch_x = torch.stack(windows, dim=0) # (batch, T, M)
    
    snn_out, _ = snn(batch_x)          # (200, 1240)
    gac_out, _ = gac_snn(batch_x)      # (200, 1240)
    
    # Transpose back to (M, T) and slice to m_max
    snn_preds = snn_out[:, :m_max].cpu().numpy().T
    gac_preds = gac_out[:, :m_max].cpu().numpy().T
    ground_truth = spike_counts[:, start_t : start_t + window_len]
    
    print(f"GT shape: {ground_truth.shape}, SNN preds: {snn_preds.shape}, GAC preds: {gac_preds.shape}")
    
    # Sort neurons by firing rate for better visualization
    sort_idx = np.argsort(ground_truth.mean(axis=1))[::-1]
    # Only keep top 50 neurons to make heatmap clear
    keep_n = min(50, m_max)
    sort_idx = sort_idx[:keep_n]
    
    gt_sorted = ground_truth[sort_idx, :]
    snn_sorted = snn_preds[sort_idx, :]
    gac_sorted = gac_preds[sort_idx, :]
    
    print("Plotting heatmaps...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    # Limit colormaps to the 95th percentile so spikes "pop"
    vmax = np.percentile(gt_sorted, 98)
    if vmax <= 0: vmax = 1.0
    
    axes[0].imshow(gt_sorted, aspect='auto', cmap='magma', interpolation='none', vmin=0, vmax=vmax)
    axes[0].set_title('Ground Truth Spikes (Validation Window)')
    axes[0].set_ylabel('Neuron (Top 50)')
    
    axes[1].imshow(snn_sorted, aspect='auto', cmap='magma', interpolation='none', vmin=0, vmax=vmax)
    axes[1].set_title(f'Standalone SNN Predicted Rates (val_r = 0.446)')
    axes[1].set_ylabel('Neuron')
    
    axes[2].imshow(gac_sorted, aspect='auto', cmap='magma', interpolation='none', vmin=0, vmax=vmax)
    axes[2].set_title(f'GAC Distillation Predicted Rates (co-BPS = -3.37)')
    axes[2].set_ylabel('Neuron')
    axes[2].set_xlabel('Time (50ms bins)')
    
    plt.tight_layout()
    plot_path = "heatmap_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {plot_path}")

if __name__ == "__main__":
    main()

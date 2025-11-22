import logging
from pathlib import Path
from typing import Dict

# import numpy as np
import torch
# from tqdm import tqdm
# import mauve

from src.metrics import compute_fd, compute_mmd
# from av_bench.utils import (unroll_dict, unroll_dict_all_keys, unroll_paired_dict,
#                             unroll_paired_dict_with_key)
from src.utils import unroll_dict

log = logging.getLogger()


@torch.inference_mode()
def evaluate(gt_audio_cache: Path, pred_audio_cache: Path) -> Dict[str, float]:


    gt_audio_cache = gt_audio_cache.expanduser()
    pred_audio_cache = pred_audio_cache.expanduser()

    # features on 16kHz
    # Vggish
    gt_vggish_features = torch.load(gt_audio_cache / 'vggish_features.pth', weights_only=True)
    pred_vggish_features = torch.load(pred_audio_cache / 'vggish_features.pth', weights_only=True)
    gt_vggish_features = unroll_dict(gt_vggish_features, cat=True).cpu()
    pred_vggish_features = unroll_dict(pred_vggish_features, cat=True).cpu()
    
    
    # features on 32kHz
    # passt
    gt_passt_features = torch.load(gt_audio_cache / 'passt_features_embed.pth', weights_only=True)
    pred_passt_features = torch.load(pred_audio_cache / 'passt_features_embed.pth', weights_only=True)
    gt_passt_features = unroll_dict(gt_passt_features).cpu()
    pred_passt_features = unroll_dict(pred_passt_features).cpu()

    # features on 48kHz
    # openl3
    gt_openl3_features = torch.load(gt_audio_cache / 'openl3.pth', weights_only=True)
    pred_openl3_features = torch.load(pred_audio_cache / 'openl3.pth', weights_only=True)
    
    gt_openl3_features = unroll_dict(gt_openl3_features)
    pred_openl3_features = unroll_dict(pred_openl3_features)
    gt_openl3_features = gt_openl3_features.reshape(-1, gt_openl3_features.shape[-1]).cpu()
    pred_openl3_features = pred_openl3_features.reshape(-1, gt_openl3_features.shape[-1]).cpu()
    
    # LAION-CLAP
    gt_clap_features = torch.load(gt_audio_cache / 'clap_laion_audio.pth', weights_only=True)
    pred_clap_features = torch.load(pred_audio_cache / 'clap_laion_audio.pth', weights_only=True)
    
    gt_clap_features = unroll_dict(gt_clap_features).cpu()
    pred_clap_features = unroll_dict(pred_clap_features).cpu()
    
    # Encodec-48kHz-stereo
    # gt_encodec_features = torch.load(gt_audio_cache / 'encodec_48khz.pth', weights_only=True)
    # pred_encodec_features = torch.load(pred_audio_cache / 'encodec_48khz.pth', weights_only=True)
    
    # gt_encodec_features = unroll_dict(gt_encodec_features)
    # gt_encodec_features = gt_encodec_features.reshape(-1, gt_encodec_features.shape[-1]).cpu()
    # pred_encodec_features = unroll_dict(pred_encodec_features)
    # pred_encodec_features = pred_encodec_features.reshape(-1, pred_encodec_features.shape[-1]).cpu()
 

    output_metrics = {}
    #### FD ####
    
    
    fd_vggish = compute_fd(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['FD-VGGish'] = fd_vggish
    
    fd_passt = compute_fd(pred_passt_features.numpy(), gt_passt_features.numpy())
    output_metrics['FD-PASST'] = fd_passt
    
    fd_openl3 = compute_fd(pred_openl3_features.numpy(), gt_openl3_features.numpy())
    output_metrics['FD-OpenL3'] = fd_openl3

    fd_lclap = compute_fd(pred_clap_features.numpy(), gt_clap_features.numpy())
    output_metrics['FD-LAION-CLAP'] = fd_lclap
    
    # fd_encodec = compute_fd(pred_encodec_features.numpy(), gt_encodec_features.numpy())
    # output_metrics['FD-EeCodec-48kHz'] = fd_encodec
    
    
    ### MMD ####
    
    mmd_vggish = compute_mmd(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['MMD-VGGish'] = mmd_vggish
    
    mmd_passt = compute_mmd(pred_passt_features.numpy(), gt_passt_features.numpy())
    output_metrics['MMD-PASST'] = mmd_passt
    
    mmd_openl3 = compute_mmd(pred_openl3_features.numpy(), gt_openl3_features.numpy())
    output_metrics['MMD-OpenL3'] = mmd_openl3
    
    mmd_lclap = compute_mmd(pred_clap_features.numpy(), gt_clap_features.numpy())
    output_metrics['MMD-LAION-CLAP'] = mmd_lclap
    
    # mmd_encodec = compute_mmd(pred_encodec_features.numpy(), gt_encodec_features.numpy())
    # output_metrics['MMD-EeCodec-48kHz'] = mmd_encodec


    return output_metrics

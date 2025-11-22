# from pathlib import Path

import laion_clap
import torch
import torch.nn as nn
# import torchaudio
from hear21passt.base import get_basic_model

import json
# from encodec import EncodecModel

# from av_bench.panns import Cnn14
from src.vggish.vggish import VGGish
import torchopenl3


class ExtractionModels(nn.Module):

    def __init__(
        self, 
        clap_model_path: str, 
        # stage1_model_config: str = _stage1_model_config, 
        # stage1_ckpt_path: str = _stage1_ckpt_path, 
    ):
        super().__init__()

        # features_list = ["2048", "logits"]
        # self.panns = Cnn14(
        #     features_list=features_list,
        #     sample_rate=16000,
        #     window_size=512,
        #     hop_size=160,
        #     mel_bins=64,
        #     fmin=50,
        #     fmax=8000,
        #     classes_num=527,
        # )

        # self.panns = self.panns.eval()
        self.vggish = VGGish(postprocess=False).eval()

        # before the prediction head
        # https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/models/passt.py#L440-L441
        self.passt_model = get_basic_model(mode="all")
        self.passt_model.eval()

        self.openl3 = torchopenl3

        enable_fusion = "fusion" in str(clap_model_path)
        self.laion_clap = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel="HTSAT-tiny").eval()
        self.laion_clap.load_ckpt(clap_model_path, verbose=False)
        self.laion_clap.eval()
        

        # self.encodec_48k = EncodecModel.encodec_model_48khz().eval().requires_grad_(False)
        # self.encodec_48k.set_target_bandwidth(24)
        
    def _get_fram_encodec(self, audio, sample_rate=48000, device='cuda'):
        with torch.no_grad():
            length = audio.shape[-1]
            duration = length / sample_rate
            assert self.encodec_48k.segment is None or duration <= 1e-5 + self.encodec_48k.segment, f"Audio is too long ({duration} > {self.encodec_48k.segment})"

            emb = self.encodec_48k.encoder(audio.to(device))
            emb = emb.transpose(-2, -1)
            return emb
        

import logging
from pathlib import Path
import os

import torch
# import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.data.audio_dataset import AudioDataset
from src.extraction_models import ExtractionModels
# import pyloudnorm as pyln   



from typing import Optional, List

log = logging.getLogger()

_clap_ckpt_path = 'path/to/ckpt/laion_clap/630k-audioset-fusion-best.pt'  # Update this path as needed

@torch.inference_mode()
def extract(
    audio_path: Path,
    output_path: Path,
    clap_model_path: Path = _clap_ckpt_path,
    *,
    csv_path: Optional[Path] = None, 
    audio_length: float = 8.0,
    batch_size: int = 128,
    num_workers: int = 32,
    device: str,
    # reconstructed_audio: bool = False,
    ):

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        col_candidates = [c for c in df.columns if c.lower() in ("audio_path")]
        if not col_candidates:
            raise ValueError("No suitable column found in CSV for audio paths.")
        audio_col = col_candidates[0]
        audio_paths: List[Path] = []
        for v in df[audio_col].astype(str):
            apath = Path(v)
            if apath.is_file():
                audio_paths.append(apath)
            else:
                log.warning(f"[SKIP] audio not found for CSV entry: {apath}")
        audios = sorted(audio_paths, key=lambda x: x.stem)

    else:
        audio_paths = []
        for root, dirs, files in os.walk(audio_path):
            for fname in files:
                if fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    audio_paths.append(Path(root) / fname)
        audios = sorted(audio_paths, key=lambda x: x.stem)
    log.info(f'{len(audios)} audio files found (recursively).')
    
    # define feature extractor
    models = ExtractionModels(clap_model_path = clap_model_path).to(device).eval()
    
    # dataloader for 16kHz. 
    dataset = AudioDataset(audios, audio_length=audio_length, sr=16_000, mono=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # extract features with VGGish
    out_dict = {}
    for wav, filename in tqdm(loader, desc="VGGish"):
        wav = wav.squeeze(1).float().to(device)
        features = models.vggish(wav).cpu()
        for i, f_name in enumerate(filename):
            out_dict[f_name] = features[i]

    output_path.mkdir(parents=True, exist_ok=True)
    vggish_feature_path = output_path / 'vggish_features.pth'
    log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
    torch.save(out_dict, vggish_feature_path)
    del out_dict
    
    
    # dataloader for 32kHz. 
    dataset = AudioDataset(audios, audio_length=audio_length, sr=32_000, mono=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # extract features with PaSST
    out_features = {}
    out_logits = {}
    for wav, filename in tqdm(loader, desc="PaSST"):
        wav = wav.squeeze(1).float().to(device)

        if (wav.size(-1) >= 320000):
            wav = wav[..., :320000]
        else:
            wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

        features = models.passt_model(wav).cpu()
        # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
        # logits is 527 dim; features is 768
        logits = features[:, :527]
        features = features[:, 527:]
        for i, f_name in enumerate(filename):
            out_features[f_name] = features[i]
            out_logits[f_name] = logits[i]
    output_path.mkdir(parents=True, exist_ok=True)
    passt_feature_path = output_path / 'passt_features_embed.pth'
    log.info(f'Saving {len(out_features)} features to {passt_feature_path}')
    torch.save(out_features, passt_feature_path)

    passt_feature_path = output_path / 'passt_logits.pth'
    log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
    torch.save(out_logits, passt_feature_path)
    del out_features, out_logits

    # dataloader for 48kHz.
    dataset = AudioDataset(audios, audio_length=audio_length, sr=48_000, mono=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # extract features with OpenL3
    out_dict = {}
    for wav, filename in tqdm(loader, desc="OpenL3"):
        wav = wav.to(device)

        # get embeddings for left and right separately
        emb, _ = models.openl3.get_audio_embedding(
            wav, sr=48000,
            content_type="env",
            input_repr="mel256",
            embedding_size=512,
            hop_size=0.5,
        )
        openl3_features = emb.cpu()
        for i, f_name in enumerate(filename):
            out_dict[f_name] = openl3_features[i]   

    output_path.mkdir(parents=True, exist_ok=True)
    openl3_features_path = output_path / 'openl3.pth'
    log.info(f'Saving {len(out_dict)} features to {openl3_features_path}')
    torch.save(out_dict, openl3_features_path)
    del out_dict
    
    # extract features with CLAP
    out_dict = {}
    for wav, filename in tqdm(loader, desc="L-CLAP"):
        wav = wav.squeeze(1).to(device) 
        clap_features = models.laion_clap.get_audio_embedding_from_data(wav, use_tensor=True)
        for i, f_name in enumerate(filename):
            out_dict[f_name] = clap_features[i] 
    clap_feature_path = output_path / 'clap_laion_audio.pth'
    log.info(f'Saving {len(out_dict)} features to {clap_feature_path}')
    torch.save(out_dict, clap_feature_path)
    del out_dict
    
    # EnCodec_48kHz features
    # out_dict = {}
    # for wav, filename in tqdm(loader, desc="EnCodec_48kHz"):
    #     wav = wav.to(device)       

    #     # get embeddings for left and right separately
    #     # move to CPU and concatenate along feature dimension
    #     encodec_segment_length = models.encodec_48k.segment_length
    #     assert wav.dim() == 3
    #     _, channels, length = wav.shape
    #     assert channels > 0 and channels <= 2
    #     stride = encodec_segment_length

    #     encoded_frames: list[torch.Tensor] = []
    #     for offset in range(0, length, stride):
    #         frame = wav[:, :, offset:offset + encodec_segment_length]
    #         encoded_frames.append(models._get_fram_encodec(frame, sample_rate=48_000, device=device))
    #     # Concatenate
    #     encoded_frames = torch.cat(encoded_frames, dim=1) # (B, T, 128)
    #     for i, f_name in enumerate(filename):
    #         out_dict[f_name] = encoded_frames[i]  # store (T, 150)

    # output_path.mkdir(parents=True, exist_ok=True)
    # encodec_48khz_features_path = output_path / 'encodec_48khz.pth'
    # log.info(f'Saving {len(out_dict)} features to {encodec_48khz_features_path}')
    # torch.save(out_dict, encodec_48khz_features_path)



# def int16_to_float32(x):
#     """
#     Convert a NumPy array of int16 values to float32.
#     Parameters:
#         x (numpy.ndarray): A NumPy array of int16 values.
#     Returns:
#         numpy.ndarray: A NumPy array of float32 values, scaled from the int16 input.
#     """

#     return (x / 32767.0).to(torch.float32)

# def float32_to_int16(x):
#     """
#     Converts a NumPy array of float32 values to int16 values.
#     This function clips the input array values to the range [-1.0, 1.0] and then scales them to the range of int16 
#     (-32768 to 32767).
#     Parameters:
#         x (numpy.ndarray): A NumPy array of float32 values to be converted.
#     Returns:
#         numpy.ndarray: A NumPy array of int16 values.
#     """

#     x = torch.clip(x, min=-1., max=1.)
#     return (x * 32767.).to(torch.int16)
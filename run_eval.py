import json
import logging
from pathlib import Path

import torch
from colorlog import ColoredFormatter

from src.args import get_eval_parser
from src.evaluate import evaluate
from src.extract import extract

log = logging.getLogger()
device = 'cuda'

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


setup_eval_logging()


@torch.inference_mode()
def main(args):
    gt_audio: Path = args.gt_audio
    gt_cache: Path = args.gt_cache
    pred_audio: Path = args.pred_audio
    pred_cache: Path = args.pred_cache
    audio_length: float = args.audio_length
    num_workers: int = args.num_workers
    gt_batch_size: int = args.gt_batch_size
    pred_batch_size: int = args.pred_batch_size
    recompute_gt_cache: bool = args.recompute_gt_cache
    recompute_pred_cache: bool = args.recompute_pred_cache
    clap_model_path: str = args.clap_model_path
    # csv_file: str = args.csv_file

    # apply default path
    if gt_cache is None:
        if gt_audio is None:
            raise ValueError('Must specify either gt_audio or gt_cache')
        gt_cache = gt_audio / 'cache'
        log.info(f'No gt cache specified, using default {gt_cache}')
        log.info(
            f'NOTE: If you are evaluating on video datasets, you must extract video cache separately'
            + f' via extract_video.py. Otherwise video-related scores will be skipped.')

    if pred_cache is None:
        if pred_audio is None:
            raise ValueError('Must specify either pred_audio or pred_cache')
        pred_cache = pred_audio / 'cache'
        log.info(f'No pred cache specified, using default {pred_cache}')
        
    # if recon_cache is None:
    #     recon_cache = recon_cache / 'cache'
    #     log.info(f'No recon_cache cache specified, using default {recon_cache}')

    gt_cache = gt_cache.expanduser()
    pred_cache = pred_cache.expanduser()

    log.info(f'GT cache path: {gt_cache}')
    log.info(f'Pred cache: {pred_cache}')
    log.info(f'Audio length: {audio_length}')

    # extract for GT if needed
    if not (gt_cache / 'clap_laion_audio.pth').exists() or recompute_gt_cache:
        log.info('Extracting GT cache...')
        if gt_audio is None:
            raise ValueError('Must specify gt_audio to compute gt_cache')
        gt_audio = gt_audio.expanduser()
        log.info(f'GT audio path: {gt_audio}')
        extract(
            audio_path=gt_audio,
            output_path=gt_cache,
            # csv_path=csv_file,
            clap_model_path=clap_model_path,
            audio_length=audio_length,
            device=device,
            batch_size=gt_batch_size,
            num_workers=num_workers,
        )
    
    # extract for pred if needed
    if not (pred_cache / 'clap_laion_audio.pth').exists() or recompute_pred_cache:
        log.info('Extracting pred cache...')
        if pred_audio is None:
            raise ValueError('Must specify pred_audio to compute pred_cache')
        pred_audio = pred_audio.expanduser()
        log.info(f'Pred audio path: {pred_audio}')
        extract(
            audio_path=pred_audio,
            output_path=pred_cache,
            audio_length=audio_length,
            device=device,
            batch_size=pred_batch_size,
            num_workers=num_workers,
            clap_model_path=clap_model_path,
            # csv_path=csv_file,
        )

    log.info('Starting evaluation...')

    output_metrics = evaluate(gt_audio_cache=gt_cache, pred_audio_cache=pred_cache)

    processed_metrics = {}
    for k, v in output_metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                item_to_log = v.item()
                item_to_save = v.item()
            else:
                item_to_log = v.tolist()
                item_to_save = v.tolist()
            processed_metrics[k] = item_to_save
        else:
            item_to_log = v
            processed_metrics[k] = v

        if isinstance(item_to_log, float):
            log.info(f'{k:<10}: {item_to_log:.10f}')
        elif isinstance(item_to_log, list):
            log.info(f'{k:<10}: {str(item_to_log)}')
        else:
            log.info(f'{k:<10}: {item_to_log}')
    # write processed metrics to file
    output_metrics_file = pred_cache / 'output_metrics.json'
    with open(output_metrics_file, 'w') as f:
        json.dump(processed_metrics, f, indent=4)
    log.info(f'Output metrics written to {output_metrics_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_eval_parser().parse_args()
    main(args)

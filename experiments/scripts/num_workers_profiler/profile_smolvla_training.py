#!/usr/bin/env python3
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, has_method

EXP = Path('/root/autodl-tmp/lerobot/outputs/libero_workers_gpu_profiler_20260627')
CONFIG_PATH = '/root/autodl-tmp/lerobot/outputs/smolvla_libero_base_direct_5epoch_bs24_56975steps_explicit_20260520_2225utc/checkpoints/045580/pretrained_model/train_config.json'
SNAP = '/root/autodl-tmp/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e'
OUT = EXP / 'profiler_nw16_bs24'
TRACE_DIR = OUT / 'torch_trace'
BATCH_SIZE = 24
NUM_WORKERS = 16
TOTAL_STEPS = 50
WARMUP_STEPS = 10
PROFILE_WAIT = 5
PROFILE_WARMUP = 5
PROFILE_ACTIVE = 30


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_cfg():
    run_out = OUT / f'train_output_{int(time.time())}'
    cli_args = [
        '--policy.repo_id=smolvla-libero-profiler-nw16',
        '--policy.pretrained_path=lerobot/smolvla_base',
        f'--dataset.root={SNAP}',
        '--dataset.streaming=false',
        '--dataset.video_backend=torchcodec',
        f'--batch_size={BATCH_SIZE}',
        f'--num_workers={NUM_WORKERS}',
        '--prefetch_factor=4',
        '--persistent_workers=true',
        f'--steps={TOTAL_STEPS}',
        '--log_freq=10',
        '--save_checkpoint=false',
        f'--save_freq={TOTAL_STEPS}',
        '--eval_freq=-1',
        '--wandb.enable=false',
        f'--output_dir={run_out}',
    ]
    cfg = TrainPipelineConfig.from_pretrained(CONFIG_PATH, cli_args=cli_args)
    cfg.validate()
    return cfg


def build_training_objects(cfg):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs], cpu=False)
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs['dataset_stats'] = dataset.meta.stats
    if processor_pretrained_path is not None:
        processor_kwargs['preprocessor_overrides'] = {
            'device_processor': {'device': accelerator.device.type},
            'normalizer_processor': {
                'stats': dataset.meta.stats,
                'features': {**policy.config.input_features, **policy.config.output_features},
                'norm_map': policy.config.normalization_mapping,
            },
            'rename_observations_processor': {'rename_map': cfg.rename_map},
        }
        postprocessor_kwargs['postprocessor_overrides'] = {
            'unnormalizer_processor': {
                'stats': dataset.meta.stats,
                'features': policy.config.output_features,
                'norm_map': policy.config.normalization_mapping,
            }
        }
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if hasattr(active_cfg, 'drop_n_last_frames'):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes['dataset_from_index'],
            dataset.meta.episodes['dataset_to_index'],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=accelerator.device.type == 'cuda',
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
    return accelerator, dataset, policy, preprocessor, optimizer, lr_scheduler, cycle(dataloader)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg()
    accelerator, dataset, policy, preprocessor, optimizer, lr_scheduler, dl_iter = build_training_objects(cfg)
    policy.train()

    rows = []
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE, repeat=1)
    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(TRACE_DIR)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(1, TOTAL_STEPS + 1):
            sync_cuda()
            t0 = time.perf_counter()
            with record_function('DataLoader'):
                batch = next(dl_iter)
            t1 = time.perf_counter()

            with record_function('CPU_Preprocessing'):
                for cam_key in dataset.meta.camera_keys:
                    if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                        batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
                batch = preprocessor(batch)
            sync_cuda()
            t2 = time.perf_counter()

            with record_function('Forward'):
                with accelerator.autocast():
                    loss, _output_dict = policy.forward(batch)
            sync_cuda()
            t3 = time.perf_counter()

            with record_function('Backward'):
                accelerator.backward(loss)
            sync_cuda()
            t4 = time.perf_counter()

            with record_function('Optimizer Step'):
                grad_clip_norm = cfg.optimizer.grad_clip_norm if cfg.optimizer is not None else 0.0
                if grad_clip_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), 'update'):
                    accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
            sync_cuda()
            t5 = time.perf_counter()

            rows.append({
                'step': step,
                'is_measurement': step > WARMUP_STEPS,
                'dataloader_s': t1 - t0,
                'cpu_preprocessing_s': t2 - t1,
                'forward_s': t3 - t2,
                'backward_s': t4 - t3,
                'optimizer_step_s': t5 - t4,
                'total_s': t5 - t0,
                'loss': float(loss.detach().cpu().item()),
                'grad_norm': float(grad_norm.detach().cpu().item() if hasattr(grad_norm, 'detach') else grad_norm),
                'gpu_memory_mib': float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else math.nan,
            })
            prof.step()
            if step % 10 == 0:
                print(f'[profile] step={step}/{TOTAL_STEPS} loss={rows[-1]["loss"]:.4f} total_s={rows[-1]["total_s"]:.3f}', flush=True)

    step_df = pd.DataFrame(rows)
    step_df.to_csv(OUT / 'profiler_step_times.csv', index=False)
    meas = step_df[step_df['is_measurement']].copy()
    segments = [
        ('DataLoader', 'dataloader_s'),
        ('CPU Preprocessing', 'cpu_preprocessing_s'),
        ('Forward', 'forward_s'),
        ('Backward', 'backward_s'),
        ('Optimizer Step', 'optimizer_step_s'),
    ]
    total_mean = meas['total_s'].mean()
    breakdown = []
    for name, col in segments:
        mean_s = meas[col].mean()
        breakdown.append({'segment': name, 'mean_s': mean_s, 'percent_total': 100.0 * mean_s / total_mean})
    pd.DataFrame(breakdown).to_csv(OUT / 'profiler_time_breakdown.csv', index=False)

    try:
        table = prof.key_averages().table(sort_by='cuda_time_total' if torch.cuda.is_available() else 'cpu_time_total', row_limit=60)
    except Exception as exc:
        table = f'Could not build profiler table: {exc}'
    (OUT / 'torch_profiler_key_averages.txt').write_text(table)

    plt.rcParams.update({
        'figure.dpi': 160,
        'savefig.dpi': 220,
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
    })
    bdf = pd.DataFrame(breakdown)
    fig, ax = plt.subplots(figsize=(7.8, 3.8), constrained_layout=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax.bar(bdf['segment'], bdf['percent_total'], color=colors)
    ax.set_ylabel('Share of step time (%)')
    ax.set_title('SmolVLA Training Time Breakdown (bs=24, num_workers=16)')
    for i, r in bdf.iterrows():
        ax.text(i, r['percent_total'] + 1.0, f"{r['mean_s']:.3f}s", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(100, float(bdf['percent_total'].max()) + 10))
    fig.savefig(OUT / 'fig_profiler_time_breakdown.png', bbox_inches='tight')
    plt.close(fig)

    meta = {
        'config_path': CONFIG_PATH,
        'dataset_root': SNAP,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'total_steps': TOTAL_STEPS,
        'warmup_steps': WARMUP_STEPS,
        'profile_wait': PROFILE_WAIT,
        'profile_warmup': PROFILE_WARMUP,
        'profile_active': PROFILE_ACTIVE,
        'mean_total_s': float(total_mean),
        'mean_samples_s': float(BATCH_SIZE / total_mean),
        'max_memory_allocated_mib': float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else None,
    }
    (OUT / 'profiler_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()

# SmolVLA Fine-Tuning Project

<p align="center">
  <a href="training_images/project_poster/CS3308_02ML_poster.pdf"><strong>View project poster (PDF)</strong></a>
</p>

This repository is a project-specific delivery based on the
[LeRobot](https://github.com/huggingface/lerobot) framework. The original
LeRobot source code is kept so that training and evaluation commands can still
use `lerobot-train`, `lerobot-eval`, and the built-in SmolVLA policy code.

The files added for this project are organized in a few top-level folders.

## Project Folders

| Folder                 | Purpose                                                                      |
| ---------------------- | ---------------------------------------------------------------------------- |
| `training_scripts/`    | Reproducible training commands for the retained models.                      |
| `pretrained_models/`   | Fine-tuned model weights and processor/config files.                         |
| `training_images/`     | Training curves grouped by task/model.                                       |
| `experiments/scripts/` | Code for batch size, num_workers, GPU utilization, and profiler experiments. |
| `experiments/results/` | Figures and CSV summaries from the ablation experiments.                     |

## Retained Models

The following model folders can be used as local LeRobot policy paths:

| Model folder                               | Description                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| `pretrained_models/libero_freeze_4epoch`   | SmolVLA-Base fine-tuned on LIBERO for 4 epochs with the vision encoder frozen.    |
| `pretrained_models/libero_unfreeze_4epoch` | SmolVLA-Base fine-tuned on LIBERO for 4 epochs with the vision encoder trainable. |
| `pretrained_models/put_objects_30epoch`    | SmolVLA-Base fine-tuned on the put-object-in-basket task for 30 epochs.           |
| `pretrained_models/stack_the_cups_30epoch` | SmolVLA-Base fine-tuned on the stack-the-cups task for 30 epochs.                 |

Each model directory contains the policy config, normalization processors, and
`model.safetensors`.

## How to Use

Install or activate a LeRobot-compatible environment first. In this repository,
the official LeRobot code is still present, so the usual LeRobot commands work.

Run a retained training configuration:

```bash
bash training_scripts/train_libero_unfreeze_4epoch.sh
bash training_scripts/train_libero_freeze_4epoch.sh
bash training_scripts/train_stack_the_cups_30epoch.sh
bash training_scripts/train_put_object_in_basket_30epoch.sh
```

Use a fine-tuned checkpoint by passing one of the folders under
`pretrained_models/` as the local policy path in your LeRobot evaluation or robot
inference command.

Run or inspect the ablation experiments:

```bash
bash experiments/scripts/batch_size_ablation/run_libero_bs_ablation.sh
bash experiments/scripts/num_workers_profiler/run_workers_ablation.sh
```

The generated experiment summaries and figures are stored in
`experiments/results/`.

## Notes

- Large model files are tracked with Git LFS. After cloning, run `git lfs pull`
  if the weight files are not downloaded automatically.
- Dataset caches and intermediate training outputs are intentionally not kept in
  this repository.
- The original LeRobot documentation remains useful for installation, hardware
  setup, dataset format, and policy API details.

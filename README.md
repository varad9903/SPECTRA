# SPECTRA

Codebase instructions:

- `BioSpec/`: re-identification backbone.
- `VAPOR/`: pose-conditioned image generator used at inference time to build additional views of each query/gallery identity.

`BioSpec/` and `VAPOR/` are dual-frameworks making SPECTRA. All paths below are relative to `SPECTRA/`.

We have tried to make the code fully reproducible, if any errors do occur, we apologize and will fix them on github.

## Installation

```bash
conda create -n spectra python=3.9
conda activate spectra
pip install -r BioSpec/requirements.txt
pip install -r VAPOR/requirements.txt
```

## Directory Setup

```
SPECTRA/
├── BioSpec/
├── VAPOR/
├── stable-diffusion-v1-5/
├── sd-vae-ft-mse/
└── checkpoints/
    ├── BioSpec/
    │   └── eva02_l_bio_best_mAP.pth
    └── VAPOR/
        ├── reference_unet.pth
        ├── denoising_unet.pth
        ├── controlnet.pth
        ├── bioq.pth
        ├── ip_adapter.pth
        ├── lora_ref_unet.pth
        └── lora_den_unet.pth
```

### BioSpec/ layout

```
BioSpec/
├── config/                 base config schema (config/defaults.py)
├── configs/                per-dataset yml configs (prcc, ltcc, Celeb_light)
├── data/                   dataset classes + dataloader
├── loss/                   loss functions
├── model/                  backbone, cross-attention refiner, GRL heads
├── processor/              train / eval loop
├── solver/                 optimizer, LR schedule
├── tools/
├── utils/
├── weights/                Qwen3-VL-32B-Thinking-FP8/, Qwen3-8B/ (see below)
├── TextCaptionDirectory/   generate_encodings.py writes here
├── generate_captions.py
├── generate_summaries.py
├── generate_encodings.py
└── train.py
```

The EVA02-CLIP-L visual encoder backbone is downloaded automatically from the Hugging Face Hub the first time `train.py` runs (via `timm`), nothing to place manually for that.

`weights/` is only needed for the captioning step below: `generate_captions.py` loads `weights/Qwen3-VL-32B-Thinking-FP8`(instruct can be used too, to avoid the thinking texts), `generate_summaries.py` loads `weights/Qwen3-8B` (overridable with `--model_path`). Both are served locally through `vllm`.

### VAPOR/ layout

```
VAPOR/
├── configs/                 inference.yaml (base/vae model paths, training hyperparameters)
├── src/                     UNets, ControlNet, attention, VAE pipeline
├── pose_library/            14 target poses x {eye_visible, eye_not_visible}
│                            (extracted from the PRCC training split by pose frequency)
├── adaptive_pose.py         skeleton extraction and rendering
├── bioq.py                  BioQ: identity feature -> cross-attention tokens
├── ip_adapter.py            IP-Adapter cross-attention injection
├── lora.py                  LoRA layers for the reference/denoising UNets
├── controlnet_pose.py       (in src/models/) pose conditioning
├── train_bioq_dataset.py    paired dataset for the training scripts below
├── prepare_bioq_training.py builds the training cache from a trained BioSpec model
├── train_v3_staged.py       stages 1-3
├── train_v3_coadapt.py      stage 4
├── inference.py             combined evaluation pipeline
└── requirements.txt
```

### Weights

1. Download:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

   Place both directly under `SPECTRA/`, next to `BioSpec/` and `VAPOR/` (`VAPOR/configs/inference.yaml` references them as `../stable-diffusion-v1-5` and `../sd-vae-ft-mse`).

2. Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/15HIer1157AwxadKbmKBX2OWK2qWKim8M?usp=sharing) and place them under `checkpoints/` as shown above.

`--vapor_ckpt_dir` and `--vapor_weights_dir` (used below) both point at `checkpoints/VAPOR`.

## Dataset Setup

```
<data_root>/
├── PRCC/prcc/rgb/{train,val,test}/<pid>/*.jpg
├── LTCC/LTCC_ReID/{train,query,test}/*.png
└── Celeb-reID-light/{train,query,gallery}/*.jpg
```

`<data_root>` is any directory holding the three datasets in the layout above. Nothing in the code assumes it sits inside `SPECTRA/`. The same value is passed to every script, as `--data_root` (`generate_captions.py`, `inference.py`) or as a trailing `DATA.ROOT <data_root>` override (`train.py`, `prepare_bioq_training.py`).

## Training BioSpec

### 1. Captions and text encodings

Requires `weights/Qwen3-VL-32B-Thinking-FP8` and `weights/Qwen3-8B` under `BioSpec/` (see BioSpec/ layout above).

```bash
cd BioSpec

python generate_captions.py --dataset prcc --gpu_id 0 --data_root <data_root>

python generate_summaries.py --dataset prcc --gpu_id 0

python generate_encodings.py --dataset prcc --gpu_id 0 \
    --encoders nomic-embed-v1.5 --combine_nonbio \
    --output_dir TextCaptionDirectory/PRCC_Qwen_Combined
```

Repeat with `--dataset ltcc` / `--dataset celeb_light` and the matching output directory (`TextCaptionDirectory/LTCC_Qwen`, `TextCaptionDirectory/Celeb_light_Qwen`; already set as `DATA.CAPTION_DIR` in the shipped configs).

### 2. Train

```bash
cd BioSpec
python train.py --config_file configs/prcc/eva02_l_bio.yml \
    DATA.ROOT <data_root> OUTPUT_DIR ./logs
```

Swap in `configs/ltcc/eva02_l_bio.yml` / `configs/Celeb_light/eva02_l_bio.yml` for the other datasets. Use the best-mAP checkpoint (`logs/.../eva02_l_bio_best_mAP.pth`) as `--biospec_weights` below.

## Training VAPOR

### 1. Build the training cache

```bash
cd VAPOR
python prepare_bioq_training.py \
    --biospec_config ../BioSpec/configs/prcc/eva02_l_bio.yml \
    --biospec_weights ../checkpoints/BioSpec/eva02_l_bio_best_mAP.pth \
    --vapor_config ./configs/inference.yaml \
    --output_dir bioq_train_cache \
    DATA.ROOT <data_root>
```

### 2. Stages 1-4

```bash
python train_v3_staged.py --stage 1 --vapor_config configs/inference.yaml \
    --cache_dir bioq_train_cache --ckpt_dir ../checkpoints/VAPOR --output_dir runs/stage1

python train_v3_staged.py --stage 2 --vapor_config configs/inference.yaml \
    --cache_dir bioq_train_cache --ckpt_dir ../checkpoints/VAPOR \
    --stage1_best runs/stage1/best --output_dir runs/stage2

python train_v3_staged.py --stage 3 --vapor_config configs/inference.yaml \
    --cache_dir bioq_train_cache --ckpt_dir ../checkpoints/VAPOR \
    --stage1_best runs/stage1/best --stage2_best runs/stage2/best \
    --lora_rank 32 --output_dir runs/stage3

python train_v3_coadapt.py --vapor_config configs/inference.yaml \
    --cache_dir bioq_train_cache --ckpt_dir ../checkpoints/VAPOR \
    --weights_dir runs/stage3/best --lora_rank 32 --output_dir runs/stage4
```

`runs/stage4/best/` contains `controlnet.pth`, `bioq.pth`, `ip_adapter.pth`, `lora_ref_unet.pth`, `lora_den_unet.pth`. Use this directory as `--vapor_weights_dir` below.

## Inference

```bash
cd VAPOR
python inference.py \
    --dataset prcc \
    --data_root <data_root> \
    --biospec_weights ../checkpoints/BioSpec/eva02_l_bio_best_mAP.pth \
    --vapor_ckpt_dir ../checkpoints/VAPOR \
    --vapor_weights_dir ../checkpoints/VAPOR \
    --output_dir inference_output/prcc
```

`--dataset` accepts `prcc`, `ltcc`, `celeb_light`. Results (Rank-1/5/10/20, mAP) are written to `inference_output/<dataset>/results.json`.

Other flags: `--k_poses`, `--alpha`, `--num_steps`, `--cfg_scale`, `--cn_scale`, `--cn_start`, `--cn_end`, `--limit_pids`, `--force`, `--shard_id`/`--num_shards`.

`--sim_threshold` admits a generated view only above that cosine similarity to the real embedding (`0.5` in the reported runs; omit to admit every view).

## Acknowledgements

Built on [MADE](https://github.com/moon-wh/MADE), [DIFFER](https://github.com/xliangp/DIFFER) and [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone).

import argparse
import glob
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIOSPEC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'BioSpec'))
sys.path.insert(0, BIOSPEC_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from config import cfg as biospec_cfg
from model import build_model as build_biospec_model

from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from adaptive_pose import extract_keypoints, render_skeleton


transform_biospec = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_vae = T.Compose([
    T.Resize((512, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def collect_prcc_train_images(data_root):
    prcc_dir = os.path.join(data_root, 'PRCC', 'prcc')
    if not os.path.isdir(os.path.join(prcc_dir, 'rgb')):
        prcc_dir = os.path.join(data_root, 'PRCC')
    train_dir = os.path.join(prcc_dir, 'rgb', 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"PRCC train directory not found: {train_dir}")
    images = []
    for person_dir in sorted(glob.glob(os.path.join(train_dir, '*'))):
        pid = os.path.basename(person_dir)
        for img_path in sorted(glob.glob(os.path.join(person_dir, '*.jpg'))):
            cam = os.path.basename(img_path)[0]
            images.append({
                'path': img_path,
                'pid': pid,
                'cam': cam,
                'key': f"{pid}_{os.path.basename(img_path)[:-4]}",
            })
    print(f"  Found {len(images)} training images from PRCC")
    return images


def extract_biospec_features(images, model, device, batch_size=32):
    features = {}
    model.eval()
    n = len(images)
    for start in tqdm(range(0, n, batch_size), desc="  Extracting BioSpec features"):
        batch_imgs, batch_keys = [], []
        for entry in images[start:start + batch_size]:
            img = Image.open(entry['path']).convert('RGB')
            batch_imgs.append(transform_biospec(img))
            batch_keys.append(entry['key'])
        batch_tensor = torch.stack(batch_imgs).to(device)
        cam_labels = torch.zeros(batch_tensor.shape[0], dtype=torch.long)
        with torch.no_grad():
            feats = model(batch_tensor, cam_label=cam_labels)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.shape[-1] > 1024:
                feats = feats[:, :1024]
        for i, key in enumerate(batch_keys):
            features[key] = feats[i].cpu().half()
    return features


def extract_vae_latents(images, vae, device, batch_size=16):
    latents = {}
    vae.eval()
    n = len(images)
    for start in tqdm(range(0, n, batch_size), desc="  Extracting VAE latents"):
        batch_imgs, batch_keys = [], []
        for entry in images[start:start + batch_size]:
            img = Image.open(entry['path']).convert('RGB')
            batch_imgs.append(transform_vae(img))
            batch_keys.append(entry['key'])
        batch_tensor = torch.stack(batch_imgs).to(device, dtype=vae.dtype)
        with torch.no_grad():
            latent = vae.encode(batch_tensor).latent_dist.mean
            latent = latent * 0.18215
        for i, key in enumerate(batch_keys):
            latents[key] = latent[i].cpu().half()
    return latents


def extract_skeletons(images, skeleton_dir, width=64, height=128):
    os.makedirs(skeleton_dir, exist_ok=True)
    skipped = 0
    for i, entry in enumerate(images):
        if (i + 1) % 200 == 0:
            print(f"    Skeleton {i+1}/{len(images)} (skipped {skipped} failed)")
        skel_path = os.path.join(skeleton_dir, f"{entry['key']}.jpg")
        if os.path.exists(skel_path):
            continue
        try:
            keypoints, scores = extract_keypoints(entry['path'])
            img = cv2.imread(entry['path'])
            orig_h, orig_w = img.shape[:2]
            kp = keypoints.copy()
            kp[:, 0] *= width / orig_w
            kp[:, 1] *= height / orig_h
            skel_img = render_skeleton(kp, width=width, height=height, scores=scores)
            skel_img.save(skel_path, quality=90)
        except Exception:
            skel_img = Image.new('RGB', (width, height), (0, 0, 0))
            skel_img.save(skel_path, quality=90)
            skipped += 1
    print(f"  Extracted {len(images) - skipped} skeletons, {skipped} failed (black fallback)")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract data for BioQ training")
    parser.add_argument("--biospec_config", type=str,
                        default="../BioSpec/configs/prcc/eva02_l_bio.yml")
    parser.add_argument("--biospec_weights", type=str, required=True,
                        help="Path to BioSpec best_mAP.pth")
    parser.add_argument("--vapor_config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--output_dir", type=str, default="bioq_train_cache")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_skeletons", action="store_true",
                        help="Skip skeleton extraction if already done")
    args, unknown = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    biospec_cfg.merge_from_file(args.biospec_config)
    if unknown:
        biospec_cfg.merge_from_list(unknown)
    biospec_cfg.freeze()

    print("=" * 60)
    print("Pre-extracting training data for BioQ")
    print("=" * 60)
    print(f"  DATA.ROOT:   {biospec_cfg.DATA.ROOT}")
    print(f"  BioSpec wts: {args.biospec_weights}")
    print(f"  Output dir:  {args.output_dir}")

    print("\n[1/4] Collecting PRCC training images...")
    images = collect_prcc_train_images(biospec_cfg.DATA.ROOT)

    cache_path = os.path.join(args.output_dir, "train_cache.pt")

    if os.path.exists(cache_path):
        print(f"\n[2/4] Cache already exists at {cache_path}, loading...")
        cache = torch.load(cache_path, map_location="cpu")
        print(f"  Loaded {len(cache['keys'])} entries")
    else:
        print("\n[2/4] Extracting BioSpec features...")
        from data import build_dataset
        original_cwd = os.getcwd()
        os.chdir(BIOSPEC_ROOT)
        try:
            dataset = build_dataset(biospec_cfg)
            num_train_pids = dataset.num_train_pids
            num_camera = dataset.num_camera
        finally:
            os.chdir(original_cwd)

        print(f"  Dataset: {biospec_cfg.DATA.DATASET} (PIDs={num_train_pids}, Cameras={num_camera})")

        biospec_model = build_biospec_model(biospec_cfg, num_train_pids, num_camera)
        state_dict = torch.load(args.biospec_weights, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        biospec_model.load_state_dict(new_state_dict, strict=False)
        biospec_model.to(device).eval()

        features = extract_biospec_features(images, biospec_model, device, args.batch_size)

        del biospec_model
        torch.cuda.empty_cache()

        print("\n[3/4] Extracting VAE latents...")
        vapor_cfg = OmegaConf.load(args.vapor_config)
        vae = AutoencoderKL.from_pretrained(vapor_cfg.vae_model_path).to(device, dtype=torch.float16)
        latents = extract_vae_latents(images, vae, device, batch_size=16)
        del vae
        torch.cuda.empty_cache()

        keys = [entry['key'] for entry in images]
        paths = [entry['path'] for entry in images]
        pids = [entry['pid'] for entry in images]
        feat_tensor = torch.stack([features[k] for k in keys])
        lat_tensor = torch.stack([latents[k] for k in keys])

        cache = {
            'keys': keys,
            'paths': paths,
            'pids': pids,
            'features': feat_tensor,
            'latents': lat_tensor,
        }
        torch.save(cache, cache_path)
        print(f"  Saved cache: {cache_path}")
        print(f"  Features: {feat_tensor.shape} ({feat_tensor.dtype})")
        print(f"  Latents:  {lat_tensor.shape} ({lat_tensor.dtype})")

    skeleton_dir = os.path.join(args.output_dir, "skeletons")
    if args.skip_skeletons:
        print("\n[4/4] Skipping skeleton extraction (--skip_skeletons)")
    else:
        print(f"\n[4/4] Extracting pose skeletons to {skeleton_dir}...")
        extract_skeletons(images, skeleton_dir)

    print(f"\n{'='*60}")
    print(f"  Pre-extraction complete!")
    print(f"  Cache:     {cache_path}")
    print(f"  Skeletons: {skeleton_dir}/")
    print(f"  Images:    {len(images)}")
    print(f"{'='*60}")
    print(f"\n  Next step: run train_v3_staged.py")


if __name__ == "__main__":
    main()

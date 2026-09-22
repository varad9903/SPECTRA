import argparse
import glob
import json
import os
import sys
import time
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

VAPOR_ROOT = os.path.dirname(os.path.abspath(__file__))
BIOSPEC_ROOT = os.path.abspath(os.path.join(VAPOR_ROOT, "..", "BioSpec"))
for p in (BIOSPEC_ROOT, VAPOR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import cfg as biospec_cfg
from model import build_model as build_biospec_model
from data import build_dataset
from utils.metrics import eval_func, eval_func_LTCC
from utils.nfc import mean_enhancement, neighbour_calibrated_enhancement

from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.controlnet_pose import ControlNetPose
from src.pipelines.pipeline_v2 import Pose2ImagePipeline_v2
from bioq import BioQ
from ip_adapter import inject_ip_adapter, load_ip_adapter_state_dict
from lora import inject_lora, load_lora_state_dict


DATASETS = {
    "prcc": {
        "config": os.path.join(BIOSPEC_ROOT, "configs", "prcc", "eva02_l_bio.yml"),
        "splits": {"gallery": "gallery", "query_same": "query_same", "query_diff": "query_diff"},
        "protocol": "prcc",
    },
    "ltcc": {
        "config": os.path.join(BIOSPEC_ROOT, "configs", "ltcc", "eva02_l_bio.yml"),
        "splits": {"gallery": "gallery", "query": "query"},
        "protocol": "ltcc",
    },
    "celeb_light": {
        "config": os.path.join(BIOSPEC_ROOT, "configs", "Celeb_light", "eva02_l_bio.yml"),
        "splits": {"gallery": "gallery", "query": "query"},
        "protocol": "celeb",
    },
}

BIOSPEC_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



COCO_18_NAMES = [
    "Nose", "Neck", "R-Shoulder", "R-Elbow", "R-Wrist",
    "L-Shoulder", "L-Elbow", "L-Wrist", "R-Hip", "R-Knee", "R-Ankle",
    "L-Hip", "L-Knee", "L-Ankle", "R-Eye", "L-Eye", "R-Ear", "L-Ear",
]
IDX_NOSE, IDX_R_EYE, IDX_L_EYE = 0, 14, 15

SKELETON_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17),
]
LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170),
]
KEYPOINT_COLORS = LIMB_COLORS + [(255, 85, 170)]


def render_skeleton(kp_normalized, width=64, height=128, scores=None, score_threshold=0.3):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kp_px = kp_normalized.copy()
    kp_px[:, 0] *= width
    kp_px[:, 1] *= height
    if scores is None:
        scores = np.ones(18, dtype=np.float32)

    for i, (a, b) in enumerate(SKELETON_LIMBS):
        if scores[a] < score_threshold or scores[b] < score_threshold:
            continue
        pt1 = (int(round(kp_px[a][0])), int(round(kp_px[a][1])))
        pt2 = (int(round(kp_px[b][0])), int(round(kp_px[b][1])))
        if not (0 <= pt1[0] < width and 0 <= pt1[1] < height
                and 0 <= pt2[0] < width and 0 <= pt2[1] < height):
            continue
        cv2.line(canvas, pt1, pt2, LIMB_COLORS[i % len(LIMB_COLORS)], 2, cv2.LINE_AA)

    for i in range(18):
        if scores[i] < score_threshold:
            continue
        pt = (int(round(kp_px[i][0])), int(round(kp_px[i][1])))
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            cv2.circle(canvas, pt, 3, KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)], -1, cv2.LINE_AA)

    return Image.fromarray(canvas)


def load_pose_set(dir_path):
    json_files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"no pose .json files found in {dir_path}")
    poses = np.zeros((len(json_files), 18, 2), dtype=np.float32)
    for i, jf in enumerate(json_files):
        with open(jf) as f:
            data = json.load(f)
        for k, name in enumerate(COCO_18_NAMES):
            if name in data["keypoints"]:
                poses[i, k, 0] = float(data["keypoints"][name]["x"])
                poses[i, k, 1] = float(data["keypoints"][name]["y"])
    return poses


_MP_TO_COCO = [
    (0, 0), (12, 2), (14, 3), (16, 4), (11, 5), (13, 6), (15, 7),
    (24, 8), (26, 9), (28, 10), (23, 11), (25, 12), (27, 13),
    (5, 14), (2, 15), (8, 16), (7, 17),
]
_MP_L_SHOULDER, _MP_R_SHOULDER = 11, 12


def extract_keypoints_and_scores(image_path):
    import mediapipe as mp

    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      enable_segmentation=False, min_detection_confidence=0.3) as pose:
        result = pose.process(img_rgb)

    if result.pose_landmarks is None:
        return None
    lm = result.pose_landmarks.landmark

    kp = np.zeros((18, 2), dtype=np.float32)
    sc = np.zeros(18, dtype=np.float32)
    for mp_idx, coco_idx in _MP_TO_COCO:
        kp[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y]
        sc[coco_idx] = lm[mp_idx].visibility
    kp[1] = [(lm[_MP_L_SHOULDER].x + lm[_MP_R_SHOULDER].x) / 2,
             (lm[_MP_L_SHOULDER].y + lm[_MP_R_SHOULDER].y) / 2]
    sc[1] = min(lm[_MP_L_SHOULDER].visibility, lm[_MP_R_SHOULDER].visibility)
    return kp, sc


def classify_eye_visibility(image_paths, threshold=0.5, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if set(cached) >= set(image_paths):
            return {p: cached[p] for p in image_paths}

    result = {}
    for p in tqdm(image_paths, desc="  eye-visibility (mediapipe)"):
        out = extract_keypoints_and_scores(p)
        if out is None:
            result[p] = "eye_visible"
            continue
        _, sc = out
        result[p] = "eye_visible" if (sc[IDX_R_EYE] > threshold or sc[IDX_L_EYE] > threshold) \
            else "eye_not_visible"

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)

    n_vis = sum(v == "eye_visible" for v in result.values())
    print(f"  eye_visible: {n_vis}  eye_not_visible: {len(result) - n_vis}")
    return result



def load_biospec(config_path, weights_path, opts, device):
    biospec_cfg.merge_from_file(config_path)
    if opts:
        biospec_cfg.merge_from_list(opts)
    biospec_cfg.freeze()

    cwd = os.getcwd()
    os.chdir(BIOSPEC_ROOT)
    try:
        dataset = build_dataset(biospec_cfg)
    finally:
        os.chdir(cwd)

    model = build_biospec_model(biospec_cfg, dataset.num_train_pids, dataset.num_camera)
    state_dict = torch.load(weights_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, dataset


@torch.no_grad()
def biospec_features(image_paths, model, device, batch_size=64, desc="biospec"):
    features = {}
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"  {desc}"):
        batch_paths = image_paths[start:start + batch_size]
        imgs = [BIOSPEC_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        feats = model(batch, cam_label=torch.zeros(batch.shape[0], dtype=torch.long))
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.shape[-1] > 1024:
            feats = feats[:, :1024]
        for i, p in enumerate(batch_paths):
            features[p] = feats[i].detach().float().cpu()
    return features



def load_vapor(vapor_config_path, base_ckpt_dir, vapor_weights_dir, lora_rank, device):
    vapor_cfg = OmegaConf.load(vapor_config_path)
    weight_dtype = torch.float16 if vapor_cfg.weight_dtype == "fp16" else torch.float32

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(VAPOR_ROOT, p))

    vae = AutoencoderKL.from_pretrained(resolve(vapor_cfg.vae_model_path)).to(device, dtype=weight_dtype)

    base_path = resolve(vapor_cfg.base_model_path)
    ckpt_dir = resolve(base_ckpt_dir)
    reference_unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(device)
    reference_unet.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "reference_unet.pth"), map_location="cpu"), strict=True)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_path, "", subfolder="unet",
        unet_additional_kwargs={"use_motion_module": False, "unet_use_temporal_attention": False},
    ).to(device)
    denoising_unet.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "denoising_unet.pth"), map_location="cpu"), strict=True)

    v2_dir = resolve(vapor_weights_dir)
    controlnet = ControlNetPose.from_unet(denoising_unet).to(device)
    controlnet.load_state_dict(
        torch.load(os.path.join(v2_dir, "controlnet.pth"), map_location="cpu"), strict=True)

    bioq = BioQ(in_dim=1024, n_tokens=32, token_dim=768).to(device)
    bioq.load_state_dict(torch.load(os.path.join(v2_dir, "bioq.pth"), map_location="cpu"))

    inject_lora(reference_unet, rank=lora_rank, dropout=0.0)
    lora_ref = os.path.join(v2_dir, "lora_ref_unet.pth")
    if os.path.exists(lora_ref):
        load_lora_state_dict(reference_unet, torch.load(lora_ref, map_location="cpu"))
    inject_lora(denoising_unet, rank=lora_rank, dropout=0.0)
    lora_den = os.path.join(v2_dir, "lora_den_unet.pth")
    if os.path.exists(lora_den):
        load_lora_state_dict(denoising_unet, torch.load(lora_den, map_location="cpu"))

    inject_ip_adapter(denoising_unet, cross_attention_dim=768)
    ip_path = os.path.join(v2_dir, "ip_adapter.pth")
    if os.path.exists(ip_path):
        load_ip_adapter_state_dict(denoising_unet, torch.load(ip_path, map_location="cpu"))

    for m in (vae, reference_unet, denoising_unet, controlnet, bioq):
        m.eval()

    sched_kwargs = OmegaConf.to_container(vapor_cfg.noise_scheduler_kwargs)
    if vapor_cfg.get("enable_zero_snr", False):
        sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing",
                            prediction_type="v_prediction")
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Pose2ImagePipeline_v2(
        vae=vae.to(dtype=torch.float32),
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        controlnet=controlnet,
        scheduler=scheduler,
    ).to(device)

    return pipe, bioq, vapor_cfg


@torch.no_grad()
def generate_k_poses(image_path, f_refined, pose_kps, seed, bioq, pipe, vapor_cfg, gen_kwargs, device):
    try:
        ref_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    could not open {image_path}: {e}")
        return []

    K = len(pose_kps)
    skeletons = [render_skeleton(pose_kps[i]) for i in range(K)]

    uncond = torch.zeros_like(f_refined)
    combined = torch.cat([uncond.expand(K, -1), f_refined.expand(K, -1)], dim=0)
    identity_tokens = bioq(combined)
    ip_tokens = identity_tokens[K:]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    output = pipe(
        identity_tokens, [ref_image] * K, skeletons,
        vapor_cfg.data.train_height, vapor_cfg.data.train_width,
        gen_kwargs["num_steps"], gen_kwargs["cfg_scale"],
        batch_size=K, generator=generator, ip_adapter_tokens=ip_tokens,
        controlnet_conditioning_scale=gen_kwargs["cn_scale"],
        control_guidance_start=gen_kwargs["cn_start"],
        control_guidance_end=gen_kwargs["cn_end"],
        latent_blur_sigma=gen_kwargs["latent_blur"],
        freeu_s1=gen_kwargs["freeu_s1"], freeu_s2=gen_kwargs["freeu_s2"],
        freeu_b1=gen_kwargs["freeu_b1"], freeu_b2=gen_kwargs["freeu_b2"],
    ).images

    generated = []
    for i in range(K):
        img_i = output[i, :, 0].permute(1, 2, 0).cpu().numpy()
        generated.append(Image.fromarray((img_i * 255).astype(np.uint8)))

    try:
        w, h = ref_image.size
        generated = [g.resize((w, h), Image.LANCZOS) if g.size != (w, h) else g for g in generated]
    except Exception:
        pass
    return generated



def feature_enhancement(f_refined, gen_feats, valid_mask, alpha=0.7,
                        mode="mean", sim_threshold=None, calib_top_k=14,
                        set_ids=None):
    if mode == "mean":
        return mean_enhancement(f_refined, gen_feats, valid_mask,
                                alpha=alpha, sim_threshold=sim_threshold)
    if mode == "neighbour_calib":
        return neighbour_calibrated_enhancement(
            f_refined, gen_feats, valid_mask, alpha=alpha,
            sim_threshold=0.5 if sim_threshold is None else sim_threshold,
            top_k=calib_top_k, set_ids=set_ids)
    raise ValueError(f"unknown enhancement mode: {mode}")


def euclidean_distance(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    dist = (torch.pow(qf, 2).sum(1, keepdim=True).expand(m, n)
            + torch.pow(gf, 2).sum(1, keepdim=True).expand(n, m).t())
    dist.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist.cpu().numpy()


def run_retrieval(qf, gf, q_pids, g_pids, q_camids, g_camids, q_clothes, g_clothes, protocol):
    distmat = euclidean_distance(qf.float(), gf.float())
    if protocol == "ltcc_cc":
        all_cmc, mAP = eval_func_LTCC(distmat, np.array(q_pids), np.array(g_pids),
                                      np.array(q_camids), np.array(g_camids),
                                      np.array(q_clothes), np.array(g_clothes))
    else:
        all_cmc, mAP = eval_func(distmat, np.array(q_pids), np.array(g_pids),
                                 np.array(q_camids), np.array(g_camids))
    return {"rank1": float(all_cmc[0]), "rank5": float(all_cmc[4]),
            "rank10": float(all_cmc[9]), "rank20": float(all_cmc[19]), "mAP": float(mAP)}


def print_row(label, res):
    print(f"    {label:<28} R-1 {res['rank1']*100:5.1f}  R-5 {res['rank5']*100:5.1f}  "
          f"R-10 {res['rank10']*100:5.1f}  mAP {res['mAP']*100:5.1f}")


def collect_splits(dataset_name, dataset_obj, limit_pids=None):
    attrs = DATASETS[dataset_name]["splits"]
    splits = {name: getattr(dataset_obj, attr) for name, attr in attrs.items()}
    if limit_pids is not None:
        keep = set(sorted({e["pid"] for e in splits["gallery"]})[:limit_pids])
        splits = {n: [e for e in es if e["pid"] in keep] for n, es in splits.items()}
    for name, entries in splits.items():
        print(f"    {name:<12s} {len(entries)} images, {len(set(e['pid'] for e in entries))} ids")
    return splits



def build_parser():
    p = argparse.ArgumentParser(description="BioSpec + VAPOR combined inference")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--data_root", required=True)

    p.add_argument("--biospec_config", default=None)
    p.add_argument("--biospec_weights", required=True)
    p.add_argument("--biospec_batch_size", type=int, default=64)

    p.add_argument("--vapor_config", default=os.path.join(VAPOR_ROOT, "configs", "inference.yaml"))
    p.add_argument("--vapor_ckpt_dir",
                   default=os.path.join(VAPOR_ROOT, "..", "checkpoints", "VAPOR"))
    p.add_argument("--vapor_weights_dir", required=True,
                   help="controlnet.pth, bioq.pth, ip_adapter.pth, lora_ref_unet.pth, lora_den_unet.pth")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--pose_library", default=os.path.join(VAPOR_ROOT, "pose_library"))
    p.add_argument("--k_poses", type=int, default=14)
    p.add_argument("--eye_threshold", type=float, default=0.5)

    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--enhance", default="neighbour_calib",
                   choices=["neighbour_calib", "mean"],
                   help="neighbour_calib removes the real-to-generated offset "
                        "before aggregation; mean is the plain aggregation")
    p.add_argument("--sim_threshold", type=float, default=None,
                   help="admit a generated view only above this cosine "
                        "similarity to the real embedding (paper runs: 0.5)")
    p.add_argument("--calib_top_k", type=int, default=14,
                   help="neighbourhood size for --enhance neighbour_calib")

    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=3.5)
    p.add_argument("--cn_scale", type=float, default=0.8)
    p.add_argument("--cn_start", type=float, default=0.0)
    p.add_argument("--cn_end", type=float, default=0.6)
    p.add_argument("--latent_blur", type=float, default=0.4)
    p.add_argument("--freeu", action="store_true", default=True)
    p.add_argument("--no_freeu", action="store_true")
    p.add_argument("--freeu_s1", type=float, default=0.9)
    p.add_argument("--freeu_s2", type=float, default=0.2)
    p.add_argument("--freeu_b1", type=float, default=1.5)
    p.add_argument("--freeu_b2", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit_pids", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p


def main():
    args = build_parser().parse_args()
    if args.no_freeu:
        args.freeu = False
    device = torch.device(args.device)

    cache_dir = os.path.join(args.output_dir, "cache")
    gen_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    ds = DATASETS[args.dataset]
    biospec_config = args.biospec_config or ds["config"]

    print(f"dataset={args.dataset}")

    print("\n[1] BioSpec + dataset splits")
    opts = list(args.opts) + ["DATA.ROOT", args.data_root]
    biospec_model, dataset_obj = load_biospec(biospec_config, args.biospec_weights, opts, device)
    splits = collect_splits(args.dataset, dataset_obj, limit_pids=args.limit_pids)

    all_entries = OrderedDict()
    for set_name, entries in splits.items():
        for e in entries:
            all_entries[f"{set_name}::{e['image_path']}"] = {**e, "set_type": set_name}
    all_paths = [e["image_path"] for e in all_entries.values()]

    print("\n[2] f_refined for all test images")
    feat_cache_path = os.path.join(cache_dir, "real_features.pt")
    real_feats = torch.load(feat_cache_path, map_location="cpu") if (
        os.path.exists(feat_cache_path) and not args.force) else {}
    missing = [p for p in all_paths if p not in real_feats]
    if missing:
        real_feats.update(biospec_features(missing, biospec_model, device, args.biospec_batch_size))
        torch.save(real_feats, feat_cache_path)

    print("\n[3] eye-visibility classification")
    eye_cache_path = os.path.join(cache_dir, "eye_visibility.json")
    eye_class = classify_eye_visibility(all_paths, threshold=args.eye_threshold, cache_path=eye_cache_path)

    print("\n[4] pose library")
    s_frontal = load_pose_set(os.path.join(args.pose_library, "eye_visible"))
    s_headless = load_pose_set(os.path.join(args.pose_library, "eye_not_visible"))
    K = min(args.k_poses, len(s_frontal), len(s_headless))
    print(f"    S_frontal={len(s_frontal)} S_headless={len(s_headless)} using K={K}")

    print("\n[5] VAPOR models")
    pipe, bioq, vapor_cfg = load_vapor(args.vapor_config, args.vapor_ckpt_dir,
                                       args.vapor_weights_dir, args.lora_rank, device)

    gen_kwargs = dict(
        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
        cn_scale=args.cn_scale, cn_start=args.cn_start, cn_end=args.cn_end,
        latent_blur=args.latent_blur,
        freeu_s1=args.freeu_s1 if args.freeu else 0.0, freeu_s2=args.freeu_s2 if args.freeu else 0.0,
        freeu_b1=args.freeu_b1 if args.freeu else 0.0, freeu_b2=args.freeu_b2 if args.freeu else 0.0,
    )

    print(f"\n[6] generate {K} poses per image + re-extract features "
          f"({len(all_entries)} images)")
    keys = list(all_entries.keys())
    my_keys = keys[args.shard_id::args.num_shards] if args.num_shards > 1 else keys

    gen_cache_path = os.path.join(cache_dir, "gen_features.pt")
    gen_feats_map, valid_map = {}, {}
    if os.path.exists(gen_cache_path) and not args.force:
        cached = torch.load(gen_cache_path, map_location="cpu")
        gen_feats_map, valid_map = cached["gen_feats_map"], cached["valid_map"]

    t0 = time.time()
    for i, key in enumerate(my_keys):
        if key in gen_feats_map and not args.force:
            continue
        entry = all_entries[key]
        path, pid, set_type = entry["image_path"], entry["pid"], entry["set_type"]
        img_key = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(gen_dir, set_type, str(pid))
        os.makedirs(out_dir, exist_ok=True)
        gen_paths = [os.path.join(out_dir, f"{img_key}_pose{k:02d}.jpg") for k in range(K)]

        if args.force or not all(os.path.exists(gp) for gp in gen_paths):
            pose_set = s_frontal if eye_class.get(path, "eye_visible") == "eye_visible" else s_headless
            f_ref = real_feats[path].unsqueeze(0).to(device)
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            eta = (len(my_keys) - i - 1) / max(rate, 1e-6) / 60
            print(f"  [{i+1}/{len(my_keys)}] {set_type}/{pid}/{img_key}  {rate:.2f} img/s  eta {eta:.0f}m")
            generated = generate_k_poses(path, f_ref, pose_set[:K], args.seed,
                                         bioq, pipe, vapor_cfg, gen_kwargs, device)
            if not generated:
                valid_map[key] = torch.zeros(K, dtype=torch.bool)
                gen_feats_map[key] = torch.zeros(K, 1024)
                continue
            for gp, img in zip(gen_paths, generated):
                img.save(gp, quality=95)

        gen_feat_dict = biospec_features(gen_paths, biospec_model, device, batch_size=K, desc=img_key)
        feats = torch.stack([gen_feat_dict.get(gp, torch.zeros(1024)) for gp in gen_paths])
        valid = torch.tensor([gp in gen_feat_dict for gp in gen_paths], dtype=torch.bool)
        gen_feats_map[key] = feats
        valid_map[key] = valid

        if (i + 1) % 20 == 0:
            torch.save({"gen_feats_map": gen_feats_map, "valid_map": valid_map}, gen_cache_path)

    torch.save({"gen_feats_map": gen_feats_map, "valid_map": valid_map}, gen_cache_path)

    print("\n[7] feature enhancement")
    ordered_real = torch.stack([real_feats[all_entries[k]["image_path"]] for k in keys])
    ordered_gen = torch.stack([gen_feats_map[k] for k in keys])
    ordered_valid = torch.stack([valid_map[k] for k in keys])
    set_ids = [k.split("::", 1)[0] for k in keys]
    enhanced = feature_enhancement(ordered_real, ordered_gen, ordered_valid,
                                   alpha=args.alpha, mode=args.enhance,
                                   sim_threshold=args.sim_threshold,
                                   calib_top_k=args.calib_top_k,
                                   set_ids=set_ids)
    baseline = F.normalize(ordered_real, dim=1)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    def gather(entries, set_name):
        idxs = [key_to_idx[f"{set_name}::{e['image_path']}"] for e in entries]
        return idxs, [e["pid"] for e in entries], [e["camid"] for e in entries], \
            [e.get("clothes_id", 0) for e in entries]

    print("\n[8] retrieval")
    results = {}

    def evaluate(q_name, q_entries, g_name, g_entries, protocol, label):
        qi, qp, qc, qcl = gather(q_entries, q_name)
        gi, gp, gc, gcl = gather(g_entries, g_name)
        r_base = run_retrieval(baseline[qi], baseline[gi], qp, gp, qc, gc, qcl, gcl, protocol)
        r_enh = run_retrieval(enhanced[qi], enhanced[gi], qp, gp, qc, gc, qcl, gcl, protocol)
        print(f"\n  {label}")
        print_row("BioSpec (f_refined)", r_base)
        print_row("SPECTRA (f_enhanced)", r_enh)
        results[label] = {"baseline": r_base, "spectra": r_enh}

    if ds["protocol"] == "prcc":
        evaluate("query_diff", splits["query_diff"], "gallery", splits["gallery"], "generic", "PRCC CC")
        evaluate("query_same", splits["query_same"], "gallery", splits["gallery"], "generic", "PRCC SC")
    elif ds["protocol"] == "ltcc":
        evaluate("query", splits["query"], "gallery", splits["gallery"], "ltcc_cc", "LTCC CC")
        evaluate("query", splits["query"], "gallery", splits["gallery"], "generic", "LTCC General")
    else:
        evaluate("query", splits["query"], "gallery", splits["gallery"], "generic", "Celeb-ReID-Light")

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"dataset": args.dataset, "alpha": args.alpha, "k_poses": K,
                   "enhance": args.enhance, "sim_threshold": args.sim_threshold,
                   "calib_top_k": args.calib_top_k,
                   "results": results}, f, indent=2)
    print(f"\nsaved {results_path}")
    print(f"generated images in {gen_dir}/")


if __name__ == "__main__":
    main()

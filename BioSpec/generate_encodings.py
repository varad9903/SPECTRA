import os
import sys
import json
import shutil
import argparse
import time
import numpy as np


ENCODER_CONFIGS = {
    'EVA02-CLIP-B-16': {
        'backend': 'open_clip',
        'open_clip_model': 'EVA02-B-16',
        'pretrained': 'merged2b_s8b_b131k',
        'dim': 512,
        'max_tokens': 77,
    },
    'EVA02-CLIP-L-14': {
        'backend': 'open_clip',
        'open_clip_model': 'EVA02-L-14',
        'pretrained': 'merged2b_s4b_b131k',
        'dim': 768,
        'max_tokens': 77,
    },
    'EVA02-CLIP-bigE-14': {
        'backend': 'open_clip',
        'open_clip_model': 'EVA02-E-14-plus',
        'pretrained': 'laion2b_s9b_b144k',
        'dim': 1024,
        'max_tokens': 77,
    },
    'nomic-embed-v1.5': {
        'backend': 'sentence_transformers',
        'model_name': 'nomic-ai/nomic-embed-text-v1.5',
        'dim': 768,
        'max_tokens': 8192,
        'task_prefix': 'search_document: ',
        'trust_remote_code': True,
    },
    'jina-clip-v1': {
        'backend': 'sentence_transformers',
        'model_name': 'jinaai/jina-clip-v1',
        'dim': 768,
        'max_tokens': 8192,
        'task_prefix': '',
        'trust_remote_code': True,
    },
}

ASPECT_NAMES = {
    0: 'biometric',
    1: 'hair/facial',
    2: 'clothing',
    3: 'pose/posture',
    4: 'behavior/activity',
    5: 'environment/setting',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text encodings from Qwen descriptions/summaries (Step 3 of 3). "
                    "Supports EVA02-CLIP (77 tokens) and long-context encoders (8K tokens)."
    )
    parser.add_argument("--dataset", type=str, choices=['prcc', 'celeb_light', 'ltcc'], default='prcc',
                        help="Which dataset to generate encodings for.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Physical GPU index to use for encoding.")
    parser.add_argument("--encoders", type=str, nargs='+',
                        default=['nomic-embed-v1.5'],
                        choices=list(ENCODER_CONFIGS.keys()),
                        help="Which encoder(s) to generate encodings for. "
                             "Default: nomic-embed-v1.5 (768-dim, 8K context).")

    parser.add_argument("--combine_nonbio", action='store_true',
                        help="Combine all non-bio aspects (1-5) into one unified description. "
                             "Produces 2 features per image (bio + combined non-bio) instead of 6. "
                             "Use with NOBIO_INDEX=['1'] in training config for single-GRL architecture.")
    parser.add_argument("--nonbio_aspects", type=int, nargs='+',
                        default=[1, 2, 3, 4, 5],
                        help="Which aspect indices to combine when --combine_nonbio is set. "
                             "Default: 1 2 3 4 5 (all non-bio aspects).")

    parser.add_argument("--descriptions_json", type=str,
                        default="qwen_train_per_image_descriptions.json",
                        help="Path to per-image descriptions JSON (output of generate_captions.py).")
    parser.add_argument("--bio_json", type=str,
                        default="qwen_train_caption_summary_biometric.json",
                        help="Path to biometric summary JSON (output of generate_summaries.py).")

    parser.add_argument("--output_dir", type=str,
                        default="TextCaptionDirectory/PRCC_Qwen",
                        help="Output directory for encoded feature files.")
    parser.add_argument("--original_caption_dir", type=str,
                        default="TextCaptionDirectory/PRCC",
                        help="Existing TextCaptionDirectory (for copying val.npz).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for text encoding. Use 32 for long-context models (nomic/jina), "
                             "256 is fine for EVA02-CLIP. Reduce further if you get OOM errors.")
    return parser.parse_args()



def load_encoder(encoder_name, device):
    cfg = ENCODER_CONFIGS[encoder_name]

    if cfg['backend'] == 'open_clip':
        return _load_open_clip_encoder(cfg, device)
    elif cfg['backend'] == 'sentence_transformers':
        return _load_st_encoder(cfg, device)
    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")


def _load_open_clip_encoder(cfg, device):
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        cfg['open_clip_model'],
        pretrained=cfg['pretrained'],
        device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(cfg['open_clip_model'])

    def encode_fn(texts, batch_size=256):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                embeddings = model.encode_text(tokens)
            all_embeddings.append(embeddings.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)

    return model, encode_fn


def _load_st_encoder(cfg, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        cfg['model_name'],
        trust_remote_code=cfg.get('trust_remote_code', False),
        device=str(device)
    )

    task_prefix = cfg.get('task_prefix', '')

    def encode_fn(texts, batch_size=256):
        if task_prefix:
            texts = [task_prefix + t for t in texts]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    return model, encode_fn



def generate_per_image_npz_6aspect(encode_fn, descriptions, batch_size, encoder_name, output_dir):
    sorted_keys = sorted(descriptions.keys())
    num_images = len(sorted_keys)

    print(f"\n  Encoding {num_images} images × 6 aspects = {num_images * 6} texts...")

    all_texts = []
    metadata = []
    for key in sorted_keys:
        aspects = descriptions[key]
        if len(aspects) != 6:
            print(f"    [WARN] {key} has {len(aspects)} aspects (expected 6), padding with empty strings")
            aspects = list(aspects)
            while len(aspects) < 6:
                aspects.append("")
        for aspect_text in aspects[:6]:
            all_texts.append(aspect_text if aspect_text else "")
        metadata.append(key)

    t0 = time.time()
    embeddings = encode_fn(all_texts, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Encoded {len(all_texts)} texts in {elapsed:.1f}s ({len(all_texts)/max(elapsed,0.01):.0f} texts/sec)")

    assert embeddings.shape[0] == num_images * 6
    embed_dim = embeddings.shape[1]
    print(f"  NPZ shape: data={embeddings.shape}, metadata={len(metadata)}, embed_dim={embed_dim}")

    encoder_dir = os.path.join(output_dir, encoder_name)
    os.makedirs(encoder_dir, exist_ok=True)
    npz_path = os.path.join(encoder_dir, "train.npz")
    np.savez(npz_path, data=embeddings, metadata=np.array(metadata))
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"  Saved: {npz_path} ({file_size_mb:.1f} MB)")

    return embed_dim



def generate_per_image_npz_combined(encode_fn, descriptions, batch_size,
                                     encoder_name, output_dir, nonbio_aspects):
    sorted_keys = sorted(descriptions.keys())
    num_images = len(sorted_keys)

    aspect_names = [ASPECT_NAMES.get(a, f'aspect_{a}') for a in nonbio_aspects]
    print(f"\n  Mode: COMBINED NON-BIO")
    print(f"  Combining aspects {nonbio_aspects} ({', '.join(aspect_names)}) into unified non-bio")
    print(f"  Encoding {num_images} images × 2 features (bio + combined non-bio)...")

    all_texts = []
    metadata = []
    token_stats = []

    for key in sorted_keys:
        aspects = descriptions[key]
        if len(aspects) < 6:
            aspects = list(aspects)
            while len(aspects) < 6:
                aspects.append("")

        bio_text = aspects[0] if aspects[0] else ""
        all_texts.append(bio_text)

        nonbio_parts = []
        for aspect_idx in nonbio_aspects:
            if aspect_idx < len(aspects) and aspects[aspect_idx]:
                nonbio_parts.append(aspects[aspect_idx].strip())
        combined_nonbio = " ".join(nonbio_parts)
        all_texts.append(combined_nonbio)
        token_stats.append(len(combined_nonbio.split()))

        metadata.append(key)

    if token_stats:
        avg_words = sum(token_stats) / len(token_stats)
        max_words = max(token_stats)
        min_words = min(token_stats)
        print(f"  Combined non-bio text stats: avg={avg_words:.0f} words, "
              f"min={min_words}, max={max_words} words")
        est_max_tokens = int(max_words * 1.3)
        print(f"  Estimated max tokens: ~{est_max_tokens} (limit: 8192)")

    t0 = time.time()
    embeddings = encode_fn(all_texts, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Encoded {len(all_texts)} texts in {elapsed:.1f}s ({len(all_texts)/max(elapsed,0.01):.0f} texts/sec)")

    assert embeddings.shape[0] == num_images * 2, \
        f"Shape mismatch: got {embeddings.shape[0]}, expected {num_images * 2}"
    embed_dim = embeddings.shape[1]
    print(f"  NPZ shape: data={embeddings.shape}, metadata={len(metadata)}, embed_dim={embed_dim}")

    encoder_dir = os.path.join(output_dir, encoder_name)
    os.makedirs(encoder_dir, exist_ok=True)
    npz_path = os.path.join(encoder_dir, "train.npz")
    np.savez(npz_path, data=embeddings, metadata=np.array(metadata))
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"  Saved: {npz_path} ({file_size_mb:.1f} MB)")

    return embed_dim


def copy_or_create_val_npz(encoder_name, embed_dim, ft_nums, output_dir, original_caption_dir):
    encoder_dir = os.path.join(output_dir, encoder_name)
    os.makedirs(encoder_dir, exist_ok=True)
    val_dst = os.path.join(encoder_dir, "val.npz")

    if os.path.exists(val_dst):
        existing = np.load(val_dst, allow_pickle=True)
        existing_ft = existing['data'].shape[0] // len(existing['metadata'])
        if existing_ft == ft_nums:
            print(f"  val.npz already exists with matching ftNums={ft_nums}: {val_dst}")
            return
        else:
            print(f"  val.npz exists but ftNums mismatch ({existing_ft} vs {ft_nums}), recreating...")

    for orig_enc in ['EVA02-CLIP-L-14', 'EVA02-CLIP-bigE-14', 'EVA02-CLIP-B-16']:
        val_src = os.path.join(original_caption_dir, orig_enc, "val.npz")
        if os.path.exists(val_src):
            orig = np.load(val_src, allow_pickle=True)
            metadata = orig['metadata']
            num_val = len(metadata)

            placeholder_data = np.zeros((num_val * ft_nums, embed_dim), dtype=np.float32)
            np.savez(val_dst, data=placeholder_data, metadata=metadata)
            file_size_mb = os.path.getsize(val_dst) / (1024 * 1024)
            print(f"  Created val.npz placeholder (ftNums={ft_nums}, dim={embed_dim}, "
                  f"{num_val} images): ({file_size_mb:.1f} MB)")
            print(f"  [NOTE] Val embeddings are zeros — re-run on val descriptions for evaluation.")
            return

    print(f"  [WARN] No original val.npz found. Val loading may crash at runtime.")
    print(f"         Copy a val.npz into {encoder_dir}/ or generate val descriptions.")



def generate_biometric_json(encode_fn, bio_data, batch_size, ft_key):
    pids = sorted(bio_data.keys())
    texts = [bio_data[pid]["text"] for pid in pids]

    print(f"\n  Encoding {len(texts)} biometric summaries...")
    t0 = time.time()
    embeddings = encode_fn(texts, batch_size=batch_size)
    print(f"  Done in {time.time()-t0:.1f}s, shape: {embeddings.shape}")

    for i, pid in enumerate(pids):
        bio_data[pid][ft_key] = [embeddings[i].tolist()]

    return bio_data


def generate_nonbio_json(encode_fn, descriptions, batch_size, ft_key, nonbio_aspects):
    sorted_keys = sorted(descriptions.keys())
    all_texts = []
    for key in sorted_keys:
        aspects = descriptions[key]
        if len(aspects) < 6:
            aspects = list(aspects)
            while len(aspects) < 6:
                aspects.append("")
        nonbio_parts = []
        for idx in nonbio_aspects:
            if idx < len(aspects) and aspects[idx]:
                nonbio_parts.append(aspects[idx].strip())
        combined = " ".join(nonbio_parts)
        all_texts.append(combined)

    print(f"\n  Encoding {len(all_texts)} per-image combined non-bio descriptions...")
    t0 = time.time()
    embeddings = encode_fn(all_texts, batch_size=batch_size)
    print(f"  Done in {time.time()-t0:.1f}s, shape: {embeddings.shape}")

    result = {}
    for i, key in enumerate(sorted_keys):
        result[key] = {
            "text": all_texts[i],
            ft_key: [embeddings[i].tolist()]
        }
    return result



def main():
    args = parse_args()

    if args.dataset == 'celeb_light':
        if args.descriptions_json == 'qwen_train_per_image_descriptions.json':
            args.descriptions_json = 'qwen_celeb_light_per_image_descriptions.json'
        if args.bio_json == 'qwen_train_caption_summary_biometric.json':
            args.bio_json = 'qwen_celeb_light_caption_summary_biometric.json'
        if args.output_dir == 'TextCaptionDirectory/PRCC_Qwen':
            args.output_dir = 'TextCaptionDirectory/Celeb_light_Qwen'
        if args.original_caption_dir == 'TextCaptionDirectory/PRCC':
            args.original_caption_dir = 'TextCaptionDirectory/Celeb-reID-light'
    elif args.dataset == 'ltcc':
        if args.descriptions_json == 'qwen_train_per_image_descriptions.json':
            args.descriptions_json = 'qwen_ltcc_per_image_descriptions.json'
        if args.bio_json == 'qwen_train_caption_summary_biometric.json':
            args.bio_json = 'qwen_ltcc_caption_summary_biometric.json'
        if args.output_dir == 'TextCaptionDirectory/PRCC_Qwen':
            args.output_dir = 'TextCaptionDirectory/LTCC_Qwen'
        if args.original_caption_dir == 'TextCaptionDirectory/PRCC':
            args.original_caption_dir = 'TextCaptionDirectory/LTCC_ReID'

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPU] Using physical GPU {args.gpu_id} -> {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    combine_mode = args.combine_nonbio
    ft_nums = 2 if combine_mode else 6
    if combine_mode:
        aspect_names = [ASPECT_NAMES.get(a, f'aspect_{a}') for a in args.nonbio_aspects]
        print(f"\n[MODE] Combined non-bio \u2192 2 features per image (bio + unified non-bio)")
        print(f"[MODE] Non-bio aspects: {args.nonbio_aspects} ({', '.join(aspect_names)})")
        print(f"[MODE] Training config: BIO_INDEX=0, NOBIO_INDEX=['1'] \u2192 1 GRL")
    else:
        print(f"\n[MODE] 6-aspect \u2192 6 features per image (original format)")
        print(f"[MODE] Training config: BIO_INDEX=0, NOBIO_INDEX=['1','2','3'] \u2192 3 GRLs")

    print(f"\n{'='*60}")
    print("Loading input files...")
    print(f"{'='*60}")

    if not os.path.exists(args.descriptions_json):
        raise FileNotFoundError(f"Descriptions not found: {args.descriptions_json}")
    with open(args.descriptions_json, "r") as f:
        descriptions = json.load(f)
    print(f"  Per-image descriptions: {len(descriptions)} images")

    if not os.path.exists(args.bio_json):
        raise FileNotFoundError(f"Biometric summaries not found: {args.bio_json}")
    with open(args.bio_json, "r") as f:
        bio_data = json.load(f)
    print(f"  Biometric summaries: {len(bio_data)} persons")

    clothes_data = {}

    os.makedirs(args.output_dir, exist_ok=True)

    desc_dst = os.path.join(args.output_dir, "train_CogVLMTextDescriptions.json")
    if not os.path.exists(desc_dst):
        shutil.copy2(args.descriptions_json, desc_dst)
        print(f"\n  Copied descriptions -> {desc_dst}")
    else:
        print(f"\n  train_CogVLMTextDescriptions.json already exists, skipping copy")

    total_t0 = time.time()

    for encoder_name in args.encoders:
        cfg = ENCODER_CONFIGS[encoder_name]
        ft_key = f"ft_{encoder_name}"

        print(f"\n{'='*60}")
        print(f"Processing encoder: {encoder_name}")
        print(f"  backend: {cfg['backend']}")
        print(f"  embed_dim: {cfg['dim']}")
        print(f"  max_tokens: {cfg['max_tokens']}")
        if cfg.get('task_prefix'):
            print(f"  task_prefix: '{cfg['task_prefix']}'")
        print(f"  npz_mode: {'combined (2 per image)' if combine_mode else '6-aspect (6 per image)'}")
        print(f"{'='*60}")

        t0 = time.time()
        print(f"\n  Loading {encoder_name}...")
        model, encode_fn = load_encoder(encoder_name, device)
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        if combine_mode:
            print(f"\n  --- Step 1/4: Per-image NPZ (COMBINED non-bio) ---")
            embed_dim = generate_per_image_npz_combined(
                encode_fn, descriptions,
                args.batch_size, encoder_name, args.output_dir,
                args.nonbio_aspects
            )
        else:
            print(f"\n  --- Step 1/4: Per-image NPZ (6-aspect) ---")
            embed_dim = generate_per_image_npz_6aspect(
                encode_fn, descriptions,
                args.batch_size, encoder_name, args.output_dir
            )
        assert embed_dim == cfg['dim'], \
            f"Dimension mismatch: got {embed_dim}, expected {cfg['dim']}"

        print(f"\n  --- Step 2/4: Val NPZ (val.npz) ---")
        copy_or_create_val_npz(encoder_name, embed_dim, ft_nums,
                               args.output_dir, args.original_caption_dir)

        print(f"\n  --- Step 3/4: Biometric summary embeddings ---")
        bio_data = generate_biometric_json(
            encode_fn, bio_data,
            args.batch_size, ft_key
        )

        print(f"\n  --- Step 4/4: Per-image non-bio embeddings ---")
        nonbio_aspects = args.nonbio_aspects if combine_mode else [1, 2, 3, 4, 5]
        clothes_data = generate_nonbio_json(
            encode_fn, descriptions,
            args.batch_size, ft_key, nonbio_aspects
        )

        del model, encode_fn
        torch.cuda.empty_cache()
        print(f"\n  \u2713 {encoder_name} complete, GPU memory freed")

    print(f"\n{'='*60}")
    print("Saving output files...")
    print(f"{'='*60}")

    bio_out_path = os.path.join(args.output_dir, "train_caption_summary_biometric.json")
    with open(bio_out_path, "w") as f:
        json.dump(bio_data, f, indent=4)
    bio_size = os.path.getsize(bio_out_path) / (1024 * 1024)
    print(f"  Biometric: {bio_out_path} ({bio_size:.1f} MB)")

    clothes_out_path = os.path.join(args.output_dir, "train_caption_summary_clothes.json")
    with open(clothes_out_path, "w") as f:
        json.dump(clothes_data, f, indent=4)
    clothes_size = os.path.getsize(clothes_out_path) / (1024 * 1024)
    print(f"  Non-bio:   {clothes_out_path} ({clothes_size:.1f} MB)")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"DONE! All encodings generated in {total_elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"NPZ mode: {'COMBINED non-bio (2 per image)' if combine_mode else '6-aspect (6 per image)'}")
    print(f"Contents:")
    for encoder_name in args.encoders:
        encoder_dir = os.path.join(args.output_dir, encoder_name)
        for f_name in ['train.npz', 'val.npz']:
            f_path = os.path.join(encoder_dir, f_name)
            if os.path.exists(f_path):
                sz = os.path.getsize(f_path) / (1024 * 1024)
                print(f"  {encoder_name}/{f_name} ({sz:.1f} MB)")
    print(f"  train_caption_summary_biometric.json ({bio_size:.1f} MB)")
    print(f"  train_caption_summary_clothes.json — non-bio descriptions ({clothes_size:.1f} MB)")
    print(f"  train_CogVLMTextDescriptions.json")

    print(f"\n{'='*60}")
    print("Training command:")
    print(f"{'='*60}")
    dataset_config_map = {'celeb_light': 'Celeb_light', 'ltcc': 'ltcc', 'prcc': 'prcc'}
    config_name = dataset_config_map.get(args.dataset, 'prcc')
    for encoder_name in args.encoders:
        cfg = ENCODER_CONFIGS[encoder_name]
        if '-' in encoder_name:
            parts = encoder_name.rsplit('-', 1)
            text_model = parts[0]
        else:
            text_model = encoder_name

        if combine_mode:
            print(f"\n  # {encoder_name} \u2014 COMBINED non-bio (1 GRL):")
            print(f"  python train.py --config_file configs/{config_name}/eva02_l_bio.yml \\")
            print(f"    DATA.ROOT DatasetsDirectory \\")
            print(f"    DATA.CAPTION_DIR {args.output_dir} \\")
            print(f"    DATA.TEXT_MODEL {text_model} \\")
            print(f"    DATA.NOBIO_INDEX \"['1']\" \\")
            print(f"    MODEL.CLIP_DIM {cfg['dim']}")
        else:
            print(f"\n  # {encoder_name} \u2014 6-aspect ({len(args.nonbio_aspects)} GRLs):")
            print(f"  python train.py --config_file configs/{config_name}/eva02_l_bio.yml \\")
            print(f"    DATA.ROOT DatasetsDirectory \\")
            print(f"    DATA.CAPTION_DIR {args.output_dir} \\")
            print(f"    DATA.TEXT_MODEL {text_model} \\")
            print(f"    MODEL.CLIP_DIM {cfg['dim']}")


if __name__ == "__main__":
    main()

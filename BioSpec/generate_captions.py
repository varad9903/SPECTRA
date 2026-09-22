import os
import json
import glob
import random
import argparse
import time
from tqdm import tqdm
import re
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')


SYSTEM_PROMPT = (
    "You are a precise image analyst. Respond ONLY with the requested description. "
    "Do NOT include any reasoning, thinking process, internal monologue, analysis steps, "
    "bullet points, numbered lists, or meta-commentary. "
    "Start your response directly with the descriptive paragraph."
)


aspect_0_prompt = (
    "Describe this person's overall physical appearance in one paragraph. "
    "Include their apparent age range, gender, body build (slender, average, stocky), "
    "and height relative to the surroundings. "
    "Do NOT mention clothing, accessories, or background details."
)

aspect_1_prompt = (
    "Describe this person's facial hair and hair features in one paragraph. "
    "Include hair color, hair style (length, texture, tied back, etc.), eye color if visible, "
    "and any distinctive facial features like freckles, scars, or tattoos. "
    "Do NOT mention clothing, accessories, or background details."
)

aspect_2_prompt = (
    "Describe this person's clothing and accessories in one paragraph. "
    "Include the color, pattern, and style of their upper garment, lower garment, and footwear. "
    "Also mention any visible accessories such as glasses, watches, jewelry, bags, or logos. "
    "Do NOT mention the person's physical traits like gender, age, height, or body build."
)

aspect_3_prompt = (
    "Describe this person's posture and gait in one paragraph. "
    "Include their body alignment (upright, leaning, slouching), walking style "
    "(brisk, slow, limping), and general demeanor suggested by their stance (confident, tired, hurried). "
    "Do NOT mention clothing or physical identity features."
)

aspect_4_prompt = (
    "Describe this person's behavior and interaction with the environment in one paragraph. "
    "Include any visible gestures, facial expressions, whether they are interacting with other people "
    "or objects, and their apparent mood or reaction to the context. "
    "Do NOT mention clothing or physical identity features."
)

aspect_5_prompt = (
    "Describe the background and environment surrounding this person in one paragraph. "
    "Include the setting (indoor, outdoor, office, street, etc.), any notable objects or elements "
    "in the vicinity, and the general atmosphere or mood of the environment. "
    "Do NOT describe the person's appearance or clothing."
)

PER_IMAGE_PROMPTS = [
    aspect_0_prompt,
    aspect_1_prompt,
    aspect_2_prompt,
    aspect_3_prompt,
    aspect_4_prompt,
    aspect_5_prompt,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Per-Image Description Generator (Step 1 of 2). "
                    "Generates 6-aspect descriptions per image. "
                    "Run generate_summaries.py afterwards to create summaries."
    )
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi",
                        help="Choose 'single' to run on one GPU, or 'multi' to split across all GPUs.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Physical GPU index for single-GPU mode (e.g. 1 uses nvidia-smi GPU 1).")
    parser.add_argument("--visible_gpus", type=str, default="0,2,3",
                        help="Comma-separated physical GPU IDs for multi-GPU mode (e.g. '0,2,3').")
    parser.add_argument("--max_pixels", type=int, default=256,
                        help="Max visual tokens. Use 256 for PRCC crops.")
    parser.add_argument("--num_persons", type=int, default=10,
                        help="Number of person IDs to process (default: 10). Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selecting person IDs.")
    parser.add_argument("--max_images_per_group", type=int, default=9999,
                        help="Max images per camera group (A/B or C).")
    parser.add_argument("--split", type=int, default=1,
                        help="Total number of splits (e.g. 2 to split across 2 GPUs).")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Which split this instance handles (0-indexed). E.g. 0 = first half, 1 = second half.")
    parser.add_argument("--dataset", type=str, choices=['prcc', 'celeb_light', 'ltcc'], default='prcc',
                        help="Dataset to generate captions for.")
    parser.add_argument("--data_root", type=str, default="DatasetsDirectory",
                        help="Dataset root directory. Same value as DATA.ROOT in the training configs.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional override for the output JSON filename.")
    return parser.parse_args()


def configure_cuda_devices(args):
    if args.mode == "multi":
        gpu_ids = [g.strip() for g in args.visible_gpus.split(",") if g.strip()]
        if not gpu_ids:
            raise ValueError("--visible_gpus must list at least one GPU id")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"\n[INIT] MULTI-GPU mode: physical GPUs [{os.environ['CUDA_VISIBLE_DEVICES']}] "
              f"-> vLLM logical cuda:0..{len(gpu_ids) - 1}, tensor_parallel_size={len(gpu_ids)}")
        return len(gpu_ids)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        
    actual_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"\n[INIT] SINGLE-GPU mode: physical GPU {actual_gpu} "
          f"(CUDA_VISIBLE_DEVICES={actual_gpu}) -> vLLM logical cuda:0")
    return 1


def _selected_physical_gpu_ids(args):
    if args.mode == "single":
        return [os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu_id)).split(",")[0]]
    return [g.strip() for g in args.visible_gpus.split(",") if g.strip()]


def audit_physical_gpus(args):
    import subprocess

    selected = _selected_physical_gpu_ids(args)
    try:
        smi_text = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        print(f"[GPU] Could not run nvidia-smi ({exc})")
        return

    print("[GPU] Physical GPU audit (nvidia-smi):")
    for line in smi_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and any(f"|  {idx} " in line for idx in selected):
            print(f"  {stripped}")

    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,fan.speed,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
    except Exception as exc:
        print(f"[GPU] Could not query GPU metrics ({exc})")
        return

    blocked = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 6:
            continue
        idx, fan, temp, util, mem_used, mem_total = parts[:6]
        if idx not in selected:
            continue
        issues = []
        if fan.upper() == "ERR!" or "ERR" in fan.upper():
            issues.append(f"fan/driver reports ERR ({fan})")
        if util.isdigit() and int(util) >= 90 and mem_used.isdigit() and int(mem_used) <= 64:
            issues.append(f"stuck at {util}% util with only {mem_used} MiB used (no real workload)")
        if issues:
            blocked.append((idx, issues))

    if blocked:
        details = "\n".join(f"  - GPU {idx}: " + "; ".join(msg) for idx, msg in blocked)
        raise RuntimeError(
            f"Selected physical GPU(s) appear unhealthy:\n{details}\n"
            "Fix (requires sudo/admin):\n"
            "  sudo nvidia-smi --gpu-reset -i <id>\n"
            "  or reboot the machine."
        )


def verify_gpu_access(args):
    import subprocess
    import torch

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        print("[GPU] CUDA-visible device status:")
        for line in out.strip().splitlines():
            print(f"  {line.strip()}")
    except Exception:
        pass

    n_visible = torch.cuda.device_count()
    if n_visible == 0:
        raise RuntimeError(
            "torch.cuda.device_count() == 0 after setting CUDA_VISIBLE_DEVICES. "
            "Selected GPUs may be busy or in ERR! state."
        )

    logical_id = 0
    try:
        free, total = torch.cuda.mem_get_info(logical_id)
        free_gib = free / (1024 ** 3)
        total_gib = total / (1024 ** 3)
        name = torch.cuda.get_device_name(logical_id)
        print(f"[GPU] logical cuda:{logical_id} OK — {name}, free {free_gib:.1f} / {total_gib:.1f} GiB")
    except Exception as exc:
        raise RuntimeError(f"Cannot access selected GPU(s): {exc}") from exc



def strip_think_block(text):
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    elif '<|end_thought|>' in text:
        text = text.rsplit('<|end_thought|>', 1)[-1]
    else:
        if '<think>' in text or '<|start_thought|>' in text:
            return ""
        reasoning_starts = [
            "Okay, the user", "Okay, let", "We are given", "We must synthesize",
            "Let me ", "Let's ", "First, I", "I need to", "I'll ",
            "Looking at the", "Now, I need", "Starting with",
            "Steps:", "Step 1", "Description 1:",
        ]
        text_lower = text.strip()[:50].lower() if text.strip() else ""
        for pattern in reasoning_starts:
            if text_lower.startswith(pattern.lower()):
                return ""

    text = text.replace('<think>', '').replace('</think>', '')
    text = text.replace('<|start_thought|>', '').replace('<|end_thought|>', '')
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.strip()
    return text



def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def generate_text_batch(prompts, image_path, max_tokens=256):
    abs_path = os.path.abspath(image_path)

    vllm_inputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file:///{abs_path}"},
                {"type": "text", "text": prompt}
            ]}
        ]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens + 4096,
        top_k=-1,
        skip_special_tokens=False,
        stop_token_ids=[],
    )

    outputs = llm.generate(vllm_inputs, sampling_params=params, use_tqdm=False)
    return [strip_think_block(out.outputs[0].text) for out in outputs]



def process_person(pid_str, image_list, train_dir, per_image_output, dataset, max_images_per_group):
    if dataset == 'prcc':
        cam_ab_images = [img for img in image_list if os.path.basename(img)[0] in ['A', 'B']]
        cam_c_images = [img for img in image_list if os.path.basename(img)[0] == 'C']
        print(f"  Person {pid_str}: {len(cam_ab_images)} images (cam A/B), {len(cam_c_images)} images (cam C)")
        if len(cam_ab_images) > max_images_per_group:
            random.seed(int(pid_str))
            cam_ab_images = random.sample(cam_ab_images, max_images_per_group)
        if len(cam_c_images) > max_images_per_group:
            random.seed(int(pid_str) + 1)
            cam_c_images = random.sample(cam_c_images, max_images_per_group)
        groups = [(cam_ab_images, "Cam A/B"), (cam_c_images, "Cam C")]
    else:
        all_imgs = sorted(image_list)
        print(f"  Person {pid_str}: {len(all_imgs)} images")
        if len(all_imgs) > max_images_per_group:
            random.seed(int(pid_str) if pid_str.isdigit() else hash(pid_str))
            all_imgs = random.sample(all_imgs, max_images_per_group)
        groups = [(all_imgs, "All")]

    new_count = 0
    for img_group, group_name in groups:
        for img_path in tqdm(img_group, desc=f"    {group_name}", leave=False):
            rel_key = img_path[len(train_dir)+1:][:-4].replace('\\', '/')
            if rel_key in per_image_output and len(per_image_output[rel_key]) == 6:
                continue
            aspects = generate_text_batch(PER_IMAGE_PROMPTS, img_path, max_tokens=1024)
            per_image_output[rel_key] = aspects
            new_count += 1

    print(f"    Generated {new_count} new descriptions, {sum(len(g) for g, _ in groups)-new_count} cached")



def safe_save_json(filepath, new_data):
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    existing.update(new_data)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=4)
    return len(existing)


def main():
    if args.output_json:
        per_image_json = args.output_json
    elif args.dataset == 'celeb_light':
        per_image_json = "qwen_celeb_light_per_image_descriptions.json"
    elif args.dataset == 'ltcc':
        per_image_json = "qwen_ltcc_per_image_descriptions.json"
    else:
        per_image_json = "qwen_train_per_image_descriptions.json"

    if args.split > 1 and not args.output_json:
        base, ext = os.path.splitext(per_image_json)
        per_image_json = f"{base}_shard{args.split_id}{ext}"
        print(f"[SHARD] Writing to shard file: {per_image_json}")


    if args.dataset == 'prcc':
        person_dirs = sorted(glob.glob(os.path.join(train_dir, "*")))
        all_pids = [os.path.basename(p) for p in person_dirs]
        pid_to_images = {os.path.basename(p): sorted(glob.glob(os.path.join(p, "*.jpg"))) for p in person_dirs}
    elif args.dataset == 'celeb_light':
        import re as _re
        all_images = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))
        print(f"[DATASET] Found {len(all_images)} .jpg images in {train_dir}")
        if len(all_images) == 0:
            raise RuntimeError(
                f"No .jpg images found in {train_dir}\n"
                f"Check that the dataset path is correct and images exist."
            )
        pattern = _re.compile(r'(\d+)_\d+_\d+')
        pid_to_images = {}
        for img_path in all_images:
            match = pattern.search(os.path.basename(img_path))
            if match:
                pid = match.group(1)
                if pid not in pid_to_images:
                    pid_to_images[pid] = []
                pid_to_images[pid].append(img_path)
        all_pids = sorted(pid_to_images.keys())
        print(f"[DATASET] Grouped into {len(all_pids)} unique person IDs")

    elif args.dataset == 'ltcc':
        import re as _re
        all_images = sorted(glob.glob(os.path.join(train_dir, "*.png")))
        print(f"[DATASET] Found {len(all_images)} .png images in {train_dir}")
        if len(all_images) == 0:
            raise RuntimeError(
                f"No .png images found in {train_dir}\n"
                f"Check that the dataset path is correct and images exist."
            )
        pattern = _re.compile(r'(\d+)_\d+_c\d+')
        pid_to_images = {}
        for img_path in all_images:
            match = pattern.search(os.path.basename(img_path))
            if match:
                pid = match.group(1)
                pid_to_images.setdefault(pid, []).append(img_path)
        all_pids = sorted(pid_to_images.keys())
        print(f"[DATASET] Grouped into {len(all_pids)} unique person IDs")


    print(f"\nTotal persons in training set: {len(all_pids)}")

    if args.num_persons > 0:
        random.seed(args.seed)
        selected_pids = random.sample(all_pids, min(args.num_persons, len(all_pids)))
    else:
        selected_pids = all_pids[:]
    selected_pids.sort()

    if args.split > 1:
        chunk_size = len(selected_pids) // args.split
        remainder = len(selected_pids) % args.split
        start = args.split_id * chunk_size + min(args.split_id, remainder)
        end = start + chunk_size + (1 if args.split_id < remainder else 0)
        selected_pids = selected_pids[start:end]
        print(f"Split {args.split_id+1}/{args.split}: processing persons {start+1}-{end} ({len(selected_pids)} persons)")

    print(f"Processing {len(selected_pids)} person IDs: {selected_pids[:10]}{'...' if len(selected_pids) > 10 else ''}")

    per_image_output = {}
    if os.path.exists(per_image_json):
        with open(per_image_json, "r") as f:
            per_image_output = json.load(f)
        sample_key = next(iter(per_image_output), None)
        if sample_key and isinstance(per_image_output[sample_key], list) and len(per_image_output[sample_key]) != 6:
            print(f"[WARN] Old format detected in {per_image_json}. Clearing stale cache.")
            per_image_output = {}
        else:
            print(f"Resumed {len(per_image_output)} existing per-image captions from {per_image_json}")

    for i, pid_str in enumerate(selected_pids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected_pids)}] Processing Person ID: {pid_str}")
        print(f"{'='*60}")

        t0 = time.time()
        process_person(pid_str, pid_to_images[pid_str], train_dir,
                       per_image_output, args.dataset, args.max_images_per_group)
        elapsed = time.time() - t0

        total = safe_save_json(per_image_json, per_image_output)
        print(f"    Saved to {per_image_json} ({total} total captions, {elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"DONE! Per-image descriptions generated for {len(selected_pids)} person IDs.")
    print(f"Output: {per_image_json}")
    if args.split > 1:
        dataset_base_names = {
            'celeb_light': 'qwen_celeb_light_per_image_descriptions',
            'ltcc': 'qwen_ltcc_per_image_descriptions',
            'prcc': 'qwen_train_per_image_descriptions',
        }
        base_name = dataset_base_names.get(args.dataset, 'qwen_train_per_image_descriptions')
        print(f"\n[SHARD] This was shard {args.split_id}/{args.split-1}.")
        print(f"[SHARD] After ALL shards finish, merge with:")
        print(f"  python -c \"")
        print(f"    import json, glob")
        print(f"    merged = {{}}")
        print(f"    for f in sorted(glob.glob('{base_name}_shard*.json')):")
        print(f"        merged.update(json.load(open(f)))")
        print(f"    json.dump(merged, open('{base_name}.json','w'), indent=2)")
        print(f"    print(f'Merged {{len(merged)}} entries into {base_name}.json')\"")
    else:
        print(f"\nNext step: run generate_summaries.py --dataset {args.dataset} to create biometric summaries.")
    print(f"{'='*60}")



if __name__ == "__main__":
    args = parse_args()
    audit_physical_gpus(args)
    tp_size = configure_cuda_devices(args)
    verify_gpu_access(args)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams

    model_path = "weights/Qwen3-VL-32B-Thinking-FP8"
    print(f"Loading weights from {model_path}...")

    min_pixels = 3136
    max_pixels_calc = args.max_pixels * 28 * 28
    print(f"Resolution locked to {args.max_pixels} max tokens ({max_pixels_calc} pixels).")
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels_calc)
    processor.tokenizer.padding_side = "left"

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        seed=args.seed
    )

    def resolve_dataset_root(data_root, primary, fallback, probe):
        root = os.path.join(data_root, *primary)
        if not os.path.isdir(root) and os.path.isdir(os.path.join(data_root, *fallback, probe)):
            root = os.path.join(data_root, *fallback)
        return root

    if args.dataset == 'prcc':
        dataset_root = resolve_dataset_root(args.data_root, ("PRCC", "prcc"), ("PRCC",), "rgb")
        train_dir = os.path.join(dataset_root, "rgb", "train")
    elif args.dataset == 'celeb_light':
        dataset_root = os.path.join(args.data_root, "Celeb-reID-light")
        train_dir = os.path.join(dataset_root, "train")
    elif args.dataset == 'ltcc':
        dataset_root = resolve_dataset_root(args.data_root, ("LTCC", "LTCC_ReID"), ("LTCC_ReID",), "train")
        train_dir = os.path.join(dataset_root, "train")

    train_dir = os.path.abspath(train_dir)
    print(f"[DATASET] dataset={args.dataset}, train_dir={train_dir}")
    if not os.path.isdir(train_dir):
        raise RuntimeError(
            f"Train directory not found: {train_dir}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Pass --data_root pointing at the dataset root (same value as DATA.ROOT)."
        )

    main()
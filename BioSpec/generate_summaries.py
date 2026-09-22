import os
import json
import glob
import random
import argparse
import time
import re

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')


SYSTEM_PROMPT = (
    "You are a precise image analyst. Respond ONLY with the requested description. "
    "Do NOT include any reasoning, thinking process, internal monologue, analysis steps, "
    "bullet points, numbered lists, or meta-commentary. "
    "Start your response directly with the descriptive paragraph."
)


bio_summary_prompt_template = (
    "You are an expert biometric analyst. Below are multiple observational descriptions of the SAME person across different camera views.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Task: Synthesize a single, definitive paragraph summarizing this person's immutable structural traits.\n"
    "Mandatory Elements: Determine the consensus on their age range, gender, relative height, core body build, and face shape.\n"
    "Rules:\n"
    "1. Find Consensus: If the descriptions slightly differ (e.g., height), output the most consistent overall profile.\n"
    "2. No Hallucinations: Only use information explicitly present in the descriptions. Do not invent details to fill gaps.\n"
    "3. CRITICAL NEGATIVE CONSTRAINT: You must completely IGNORE and EXCLUDE any mention of hair (style, color, length, facial hair), clothing, bags, or accessories.\n"
    "Output ONLY the final synthesized paragraph, with no introductory text or meta-commentary."
)




def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate biometric summaries from per-image descriptions (Step 2 of 2). "
                    "Uses Qwen3-8B (text-only) for fast, clean summaries."
    )
    parser.add_argument("--dataset", type=str, choices=['prcc', 'celeb_light', 'ltcc'], default='prcc',
                        help="Which dataset to generate summaries for.")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single",
                        help="Choose 'single' to run on one GPU, or 'multi' to split across GPUs.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Physical GPU index for single-GPU mode.")
    parser.add_argument("--visible_gpus", type=str, default="0,2,3",
                        help="Comma-separated physical GPU IDs for multi-GPU mode.")
    parser.add_argument("--num_persons", type=int, default=0,
                        help="Number of person IDs to process (0 = all available in descriptions JSON).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selecting person IDs.")
    parser.add_argument("--max_summary_descs", type=int, default=15,
                        help="Max per-image descriptions to feed into each summarization prompt.")
    parser.add_argument("--model_path", type=str, default="weights/Qwen3-8B",
                        help="Path to Qwen3-8B model weights.")
    parser.add_argument("--descriptions_json", type=str, default="qwen_train_per_image_descriptions.json",
                        help="Path to per-image descriptions JSON (output of generate_captions.py).")
    parser.add_argument("--bio_json", type=str, default="qwen_train_caption_summary_biometric.json",
                        help="Output path for biometric summaries.")
    return parser.parse_args()



def configure_cuda_devices(args):
    if args.mode == "multi":
        gpu_ids = [g.strip() for g in args.visible_gpus.split(",") if g.strip()]
        if not gpu_ids:
            raise ValueError("--visible_gpus must list at least one GPU id")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"\n[INIT] MULTI-GPU mode: physical GPUs [{os.environ['CUDA_VISIBLE_DEVICES']}] "
              f"-> tensor_parallel_size={len(gpu_ids)}")
        return len(gpu_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"\n[INIT] SINGLE-GPU mode: physical GPU {args.gpu_id}")
    return 1


def verify_gpu_access():
    import subprocess
    import torch
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        print("[GPU] CUDA-visible device status:")
        for line in out.strip().splitlines():
            print(f"  {line.strip()}")
    except Exception:
        pass
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No GPUs visible after setting CUDA_VISIBLE_DEVICES.")
    free, total = torch.cuda.mem_get_info(0)
    name = torch.cuda.get_device_name(0)
    print(f"[GPU] cuda:0 OK — {name}, free {free/(1024**3):.1f} / {total/(1024**3):.1f} GiB")



def generate_summary(prompt_text, max_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    vllm_input = {'prompt': text}

    params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([vllm_input], sampling_params=params, use_tqdm=False)
    result = outputs[0].outputs[0].text.strip()

    result = result.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

    return result



def get_person_ids_from_descriptions(per_image_output, dataset):
    pids = set()
    for key in per_image_output:
        if dataset in ('celeb_light', 'ltcc'):
            pid = key.split('_')[0]
        else:
            pid = key.split('/')[0]
        pids.add(pid)
    return sorted(pids)


def get_bio_descriptions_for_person(pid, per_image_output, dataset):
    bio_descs = []
    for key, aspects in per_image_output.items():
        if dataset in ('celeb_light', 'ltcc'):
            key_pid = key.split('_')[0]
        else:
            key_pid = key.split('/')[0]
        if key_pid != pid:
            continue
        if len(aspects) < 1:
            continue
        bio_text = aspects[0]
        if bio_text:
            bio_descs.append(bio_text)
    return bio_descs


def process_person_summaries(pid_str, per_image_output, max_sum, dataset):
    bio_descs = get_bio_descriptions_for_person(pid_str, per_image_output, dataset)
    print(f"  Person {pid_str}: {len(bio_descs)} bio descriptions")

    if len(bio_descs) == 0:
        print(f"    [SKIP] No descriptions found for person {pid_str}")
        return None

    if len(bio_descs) > max_sum:
        random.seed(int(pid_str) if pid_str.isdigit() else hash(pid_str))
        bio_descs = random.sample(bio_descs, max_sum)

    descs_text = "\n---\n".join(bio_descs)
    print(f"    Biometric summary ({len(bio_descs)} descs)...", flush=True)
    t0 = time.time()
    bio_summary = generate_summary(bio_summary_prompt_template.format(descriptions=descs_text))
    print(f"    Done in {time.time()-t0:.1f}s", flush=True)

    return {"text": bio_summary}



def main():
    if not hasattr(args, '_defaults_set'):
        if args.dataset == 'celeb_light':
            if args.descriptions_json == 'qwen_train_per_image_descriptions.json':
                args.descriptions_json = 'qwen_celeb_light_per_image_descriptions.json'
            if args.bio_json == 'qwen_train_caption_summary_biometric.json':
                args.bio_json = 'qwen_celeb_light_caption_summary_biometric.json'
        elif args.dataset == 'ltcc':
            if args.descriptions_json == 'qwen_train_per_image_descriptions.json':
                args.descriptions_json = 'qwen_ltcc_per_image_descriptions.json'
            if args.bio_json == 'qwen_train_caption_summary_biometric.json':
                args.bio_json = 'qwen_ltcc_caption_summary_biometric.json'

    if not os.path.exists(args.descriptions_json):
        raise FileNotFoundError(
            f"Descriptions file not found: {args.descriptions_json}\n"
            f"Run generate_captions.py --dataset {args.dataset} first."
        )

    with open(args.descriptions_json, "r") as f:
        per_image_output = json.load(f)
    print(f"\nLoaded {len(per_image_output)} per-image descriptions from {args.descriptions_json}")

    all_pids = get_person_ids_from_descriptions(per_image_output, args.dataset)
    print(f"Found {len(all_pids)} unique person IDs in descriptions")

    if args.num_persons > 0 and args.num_persons < len(all_pids):
        random.seed(args.seed)
        selected_pids = random.sample(all_pids, args.num_persons)
        selected_pids.sort()
    else:
        selected_pids = all_pids

    print(f"Processing {len(selected_pids)} person IDs: {selected_pids[:20]}{'...' if len(selected_pids) > 20 else ''}")

    bio_output = {}
    if os.path.exists(args.bio_json):
        with open(args.bio_json, "r") as f:
            bio_output = json.load(f)
        print(f"Resumed {len(bio_output)} existing biometric summaries")

    total_time = 0
    processed = 0

    for pid_str in selected_pids:
        pid_key = str(int(pid_str)) if pid_str.isdigit() else pid_str

        if pid_key in bio_output:
            print(f"\n  Skipping Person ID {pid_str} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Person ID: {pid_str}")
        print(f"{'='*60}")

        t_person = time.time()
        bio_result = process_person_summaries(
            pid_str, per_image_output, args.max_summary_descs, args.dataset
        )

        if bio_result is None:
            continue

        bio_output[pid_key] = bio_result

        with open(args.bio_json, "w") as f:
            json.dump(bio_output, f, indent=4)

        elapsed = time.time() - t_person
        total_time += elapsed
        processed += 1

        print(f"\n    Bio: {bio_result['text'][:150]}...")
        print(f"    Person {pid_str} done in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE! Generated bio summaries for {processed} person IDs in {total_time:.0f}s total.")
    print(f"Output: {args.bio_json} ({len(bio_output)} biometric summaries)")
    print(f"\nNext step: python generate_encodings.py --dataset {args.dataset}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    tp_size = configure_cuda_devices(args)
    verify_gpu_access()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.model_path
    print(f"Loading weights from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        seed=args.seed
    )

    main()

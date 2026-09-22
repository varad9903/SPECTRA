import numpy as np
import cv2
import os
import json
import math
import random
from PIL import Image



SKELETON_LIMBS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (0, 15),
    (14, 16),
    (15, 17),
]

LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]

KEYPOINT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]


def extract_keypoints_mediapipe(image_path):
    import mediapipe as mp
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    landmarks = None
    
    try:
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose:
            results = pose.process(img_rgb)
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            print(f"  MediaPipe (legacy API) detected pose successfully")
    except AttributeError:
        pass
    
    if landmarks is None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_heavy.task')
            if not os.path.exists(model_path):
                print(f"  Downloading MediaPipe Pose model (~30MB)...")
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                urllib.request.urlretrieve(url, model_path)
                print(f"  Downloaded to {model_path}")
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
            )
            
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = landmarker.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                print(f"  MediaPipe (Tasks API) detected pose successfully")
        except Exception as e:
            print(f"  MediaPipe Tasks API also failed: {e}")
    
    if landmarks is None:
        print(f"  WARNING: MediaPipe could not detect a pose in {os.path.basename(image_path)}")
        return _estimate_keypoints_heuristic(image_path)
    
    lm = landmarks
    
    MP_NOSE, MP_L_EYE, MP_R_EYE = 0, 2, 5
    MP_L_EAR, MP_R_EAR = 7, 8
    MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
    MP_L_ELBOW, MP_R_ELBOW = 13, 14
    MP_L_WRIST, MP_R_WRIST = 15, 16
    MP_L_HIP, MP_R_HIP = 23, 24
    MP_L_KNEE, MP_R_KNEE = 25, 26
    MP_L_ANKLE, MP_R_ANKLE = 27, 28
    
    def _lm_to_px(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)
    
    def _lm_vis(idx):
        return getattr(lm[idx], 'visibility', 1.0)
    
    kp = np.zeros((18, 2), dtype=np.float32)
    scores = np.zeros(18, dtype=np.float32)
    
    kp[0]  = _lm_to_px(MP_NOSE);        scores[0]  = _lm_vis(MP_NOSE)
    kp[1]  = (_lm_to_px(MP_L_SHOULDER) + _lm_to_px(MP_R_SHOULDER)) / 2.0
    scores[1] = min(_lm_vis(MP_L_SHOULDER), _lm_vis(MP_R_SHOULDER))
    kp[2]  = _lm_to_px(MP_R_SHOULDER);  scores[2]  = _lm_vis(MP_R_SHOULDER)
    kp[3]  = _lm_to_px(MP_R_ELBOW);     scores[3]  = _lm_vis(MP_R_ELBOW)
    kp[4]  = _lm_to_px(MP_R_WRIST);     scores[4]  = _lm_vis(MP_R_WRIST)
    kp[5]  = _lm_to_px(MP_L_SHOULDER);  scores[5]  = _lm_vis(MP_L_SHOULDER)
    kp[6]  = _lm_to_px(MP_L_ELBOW);     scores[6]  = _lm_vis(MP_L_ELBOW)
    kp[7]  = _lm_to_px(MP_L_WRIST);     scores[7]  = _lm_vis(MP_L_WRIST)
    kp[8]  = _lm_to_px(MP_R_HIP);       scores[8]  = _lm_vis(MP_R_HIP)
    kp[9]  = _lm_to_px(MP_R_KNEE);      scores[9]  = _lm_vis(MP_R_KNEE)
    kp[10] = _lm_to_px(MP_R_ANKLE);     scores[10] = _lm_vis(MP_R_ANKLE)
    kp[11] = _lm_to_px(MP_L_HIP);       scores[11] = _lm_vis(MP_L_HIP)
    kp[12] = _lm_to_px(MP_L_KNEE);      scores[12] = _lm_vis(MP_L_KNEE)
    kp[13] = _lm_to_px(MP_L_ANKLE);     scores[13] = _lm_vis(MP_L_ANKLE)
    kp[14] = _lm_to_px(MP_R_EYE);       scores[14] = _lm_vis(MP_R_EYE)
    kp[15] = _lm_to_px(MP_L_EYE);       scores[15] = _lm_vis(MP_L_EYE)
    kp[16] = _lm_to_px(MP_R_EAR);       scores[16] = _lm_vis(MP_R_EAR)
    kp[17] = _lm_to_px(MP_L_EAR);       scores[17] = _lm_vis(MP_L_EAR)
    
    n_detected = np.sum(scores > 0.3)
    print(f"  Detected {n_detected}/18 keypoints (confidence > 0.3)")
    
    return kp, scores


def extract_keypoints(image_path):
    try:
        import mediapipe
        print(f"  Using MediaPipe Pose for keypoint extraction")
        return extract_keypoints_mediapipe(image_path)
    except ImportError:
        pass
    
    try:
        from dwpose_utils import DWposeDetector
        print(f"  Using DWpose for keypoint extraction")
        detector = DWposeDetector()
        img = cv2.imread(image_path)
        pose_result, _ = detector(img)
        keypoints = pose_result['bodies']['candidate'][:18, :2]
        scores = pose_result['bodies']['candidate'][:18, 2] if pose_result['bodies']['candidate'].shape[1] > 2 else np.ones(18)
        return keypoints, scores
    except (ImportError, Exception):
        pass
    
    print(f"  WARNING: No pose detector available! Install mediapipe: pip install mediapipe")
    print(f"           Using heuristic skeleton (generic standing pose — NOT from the actual image)")
    return _estimate_keypoints_heuristic(image_path)


def _estimate_keypoints_heuristic(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    cx = w * 0.5
    
    kp = np.zeros((18, 2), dtype=np.float32)
    
    kp[0]  = [cx, h * 0.08]
    kp[14] = [cx - w * 0.03, h * 0.06]
    kp[15] = [cx + w * 0.03, h * 0.06]
    kp[16] = [cx - w * 0.06, h * 0.07]
    kp[17] = [cx + w * 0.06, h * 0.07]
    kp[1]  = [cx, h * 0.18]
    kp[2]  = [cx - w * 0.15, h * 0.20]
    kp[5]  = [cx + w * 0.15, h * 0.20]
    kp[8]  = [cx - w * 0.08, h * 0.48]
    kp[11] = [cx + w * 0.08, h * 0.48]
    kp[3]  = [cx - w * 0.20, h * 0.35]
    kp[4]  = [cx - w * 0.18, h * 0.48]
    kp[6]  = [cx + w * 0.20, h * 0.35]
    kp[7]  = [cx + w * 0.18, h * 0.48]
    kp[9]  = [cx - w * 0.10, h * 0.68]
    kp[10] = [cx - w * 0.10, h * 0.90]
    kp[12] = [cx + w * 0.10, h * 0.68]
    kp[13] = [cx + w * 0.10, h * 0.90]
    
    scores = np.ones(18, dtype=np.float32)
    return kp, scores


def _rotate_point_around_pivot(point, pivot, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    new_x = pivot[0] + dx * cos_a - dy * sin_a
    new_y = pivot[1] + dx * sin_a + dy * cos_a
    return np.array([new_x, new_y], dtype=np.float32)


def _translate_point(point, dx, dy):
    return np.array([point[0] + dx, point[1] + dy], dtype=np.float32)



def load_discovered_poses(pose_json_path=None, n_poses=9):
    if pose_json_path is None:
        pose_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pose_summary.json'
        )

    if not os.path.exists(pose_json_path):
        raise FileNotFoundError(
            f"pose_summary.json not found at {pose_json_path}\n"
            f"Run scan_dataset_poses.py first to discover poses from PRCC."
        )

    with open(pose_json_path, 'r') as f:
        data = json.load(f)

    centroids = []
    percentages = []

    for cluster in data['clusters'][:n_poses]:
        kp = np.array(cluster['centroid'], dtype=np.float32)
        centroids.append(kp)
        percentages.append(cluster['percentage'])

    total_coverage = sum(percentages)
    print(f"  Loaded {len(centroids)} discovered poses "
          f"(covering {total_coverage:.1f}% of dataset)")

    return centroids, percentages


def render_discovered_pose(centroid_norm, width=64, height=128):
    kp_px = centroid_norm.copy()
    kp_px[:, 0] *= width
    kp_px[:, 1] *= height

    return render_skeleton(kp_px, width=width, height=height)


def perturb_keypoints(keypoints, n=9, seed=42):
    centroids, _ = load_discovered_poses(n_poses=n)
    result = []
    for c in centroids:
        kp = c.copy()
        kp[:, 0] *= 64
        kp[:, 1] *= 128
        result.append(kp)
    return result[:n]


def perturb_keypoints_diverse(keypoints, n=9, seed=42):
    return perturb_keypoints(keypoints, n=n, seed=seed)


def render_skeleton(keypoints, width=64, height=128, scores=None, 
                    limb_thickness=2, point_radius=3):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    if scores is None:
        scores = np.ones(18, dtype=np.float32)
    
    for limb_idx, (i, j) in enumerate(SKELETON_LIMBS):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if scores[i] < 0.3 or scores[j] < 0.3:
            continue
            
        pt1 = (int(round(keypoints[i][0])), int(round(keypoints[i][1])))
        pt2 = (int(round(keypoints[j][0])), int(round(keypoints[j][1])))
        
        if (pt1[0] < 0 or pt1[0] >= width or pt1[1] < 0 or pt1[1] >= height or
            pt2[0] < 0 or pt2[0] >= width or pt2[1] < 0 or pt2[1] >= height):
            continue
        
        color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
        cv2.line(canvas, pt1, pt2, color, limb_thickness, lineType=cv2.LINE_AA)
    
    for kp_idx in range(min(len(keypoints), 18)):
        if scores[kp_idx] < 0.3:
            continue
        pt = (int(round(keypoints[kp_idx][0])), int(round(keypoints[kp_idx][1])))
        if pt[0] < 0 or pt[0] >= width or pt[1] < 0 or pt[1] >= height:
            continue
        color = KEYPOINT_COLORS[kp_idx % len(KEYPOINT_COLORS)]
        cv2.circle(canvas, pt, point_radius, color, -1, lineType=cv2.LINE_AA)
    
    return Image.fromarray(canvas)


def generate_adaptive_poses(image_path, n_poses=9, output_dir=None, seed=42,
                             pose_json_path=None):
    print(f"  Extracting keypoints from {os.path.basename(image_path)}...")
    keypoints, scores = extract_keypoints(image_path)

    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    scale_x = 64.0 / orig_w
    scale_y = 128.0 / orig_h

    kp_normalized = keypoints.copy()
    kp_normalized[:, 0] *= scale_x
    kp_normalized[:, 1] *= scale_y

    centroids, percentages = load_discovered_poses(
        pose_json_path=pose_json_path, n_poses=n_poses
    )

    skeleton_images = []
    all_keypoints = [kp_normalized]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx, centroid in enumerate(centroids):
        skel_img = render_discovered_pose(centroid, width=64, height=128)
        skeleton_images.append(skel_img)

        kp_px = centroid.copy()
        kp_px[:, 0] *= 64
        kp_px[:, 1] *= 128
        all_keypoints.append(kp_px)

        if output_dir:
            skel_img.save(os.path.join(output_dir,
                          f"discovered_pose_{idx+1:02d}_{percentages[idx]:.1f}pct.jpg"))

    if output_dir:
        orig_skel = render_skeleton(kp_normalized, width=64, height=128, scores=scores)
        orig_skel.save(os.path.join(output_dir, "detected_pose_original.jpg"))

    print(f"  Generated {len(skeleton_images)} discovered-pose skeletons")
    print(f"  Coverage: {sum(percentages):.1f}% of PRCC dataset pose distribution")
    return skeleton_images, all_keypoints


if __name__ == "__main__":
    """Test the adaptive pose module standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate discovered-pose skeletons")
    parser.add_argument("--image", type=str, required=True, help="Input person image")
    parser.add_argument("--n_poses", type=int, default=9,
                        help="Number of discovered poses (default 9)")
    parser.add_argument("--output_dir", type=str, default="adaptive_poses_output")
    parser.add_argument("--pose_json", type=str, default=None,
                        help="Path to pose_summary.json (auto-detected if omitted)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    skeletons, keypoints = generate_adaptive_poses(
        args.image, n_poses=args.n_poses, output_dir=args.output_dir,
        seed=args.seed, pose_json_path=args.pose_json
    )

    print(f"\nSaved {len(skeletons)} discovered-pose skeletons to {args.output_dir}/")

    n = len(skeletons)
    cols = min(n + 1, 10)
    rows_needed = (n + 1 + cols - 1) // cols
    grid_w = 64 * cols
    grid_h = 128 * rows_needed
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

    orig_skel = render_skeleton(keypoints[0], width=64, height=128)
    grid.paste(orig_skel, (0, 0))

    for i, skel in enumerate(skeletons):
        col = (i + 1) % cols
        row = (i + 1) // cols
        grid.paste(skel, (col * 64, row * 128))

    grid_path = os.path.join(args.output_dir, "all_poses_grid.jpg")
    grid.save(grid_path, quality=95)
    print(f"Grid saved: {grid_path}")
    print(f"\nFirst cell = detected pose from input image")
    print(f"Remaining = top {n} discovered poses from PRCC dataset")

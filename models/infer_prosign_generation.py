import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

import utils.pose_normalization as pn

from transformers import T5Tokenizer

from models.model_prosign_generation import ProSign_generation, T5_MODEL_ID
from models.model_prosign_generation_base import use_local_t5_files_only

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None


def get_args_parser():
    parser = argparse.ArgumentParser("ProSign_generation inference")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--pose-dirname", default="pose", type=str)
    parser.add_argument("--overwrite-pose", action="store_true")
    parser.add_argument("--max-new-tokens", default=150, type=int)
    parser.add_argument("--num-beams", default=2, type=int)
    return parser


def build_pose_normalizer():
    """Builds the evaluation-time keypoint normalization pipeline."""

    return pn.Compose(
        [
            pn.RemoveFace(),
            pn.KeypointLinearInterpolation(),
            pn.ScaleNormalize(scale_factor=1, frame_level=False),
            pn.CenterNormalize(frame_level=False),
            pn.AddRelativeVelocity(),
            pn.NumpyToTensor(),
        ]
    )


def extract_pose_to_pkl(video_path, pose_path):
    """Runs MediaPipe Holistic on a video and saves the extracted pose sequence."""

    if cv2 is None:
        raise ImportError("opencv-python is required for video inference.")
    if mp is None:
        raise ImportError("mediapipe is required for pose extraction.")

    pose_path.parent.mkdir(parents=True, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    keypoints_list = []
    confidences_list = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                frame_keypoints, frame_confidences = extract_frame_keypoints(results)
                keypoints_list.append(frame_keypoints)
                confidences_list.append(frame_confidences)
        finally:
            capture.release()

    if not keypoints_list:
        raise RuntimeError(f"No frames were extracted from video: {video_path}")

    pose_data = {
        "keypoints": np.stack(keypoints_list).astype(np.float32),
        "confidences": np.stack(confidences_list).astype(np.float32),
    }

    with pose_path.open("wb") as file:
        pickle.dump(pose_data, file)

    return pose_data


def extract_frame_keypoints(results):
    """Extracts 75 keypoints from a single MediaPipe Holistic result."""

    pose_coords, pose_confidences = extract_landmark_block(
        landmarks=getattr(results, "pose_landmarks", None),
        expected_count=33,
        use_visibility=True,
    )
    left_hand_coords, left_hand_confidences = extract_landmark_block(
        landmarks=getattr(results, "left_hand_landmarks", None),
        expected_count=21,
        use_visibility=False,
    )
    right_hand_coords, right_hand_confidences = extract_landmark_block(
        landmarks=getattr(results, "right_hand_landmarks", None),
        expected_count=21,
        use_visibility=False,
    )

    keypoints = np.concatenate([pose_coords, left_hand_coords, right_hand_coords], axis=0)
    confidences = np.concatenate([pose_confidences, left_hand_confidences, right_hand_confidences], axis=0)
    return keypoints, confidences


def extract_landmark_block(landmarks, expected_count, use_visibility):
    """Converts a landmark list into coordinates and per-joint confidences."""

    coords = np.zeros((expected_count, 3), dtype=np.float32)
    confidences = np.zeros(expected_count, dtype=np.float32)

    if landmarks is None:
        return coords, confidences

    for index, landmark in enumerate(landmarks.landmark[:expected_count]):
        coords[index] = [landmark.x, landmark.y, landmark.z]
        if use_visibility:
            confidences[index] = float(getattr(landmark, "visibility", 1.0))
        else:
            confidences[index] = 1.0

    return coords, confidences


def load_pose_data(video_path, pose_dirname, overwrite_pose):
    """Loads an existing pose pickle or creates one next to the video."""

    video_path = Path(video_path)
    pose_path = video_path.parent / pose_dirname / f"{video_path.stem}.pkl"

    if pose_path.exists() and not overwrite_pose:
        with pose_path.open("rb") as file:
            pose_data = pickle.load(file)
        return pose_data, pose_path

    pose_data = extract_pose_to_pkl(video_path, pose_path)
    return pose_data, pose_path


def build_src_input(pose_data, sample_name):
    """Builds the model input batch from a pose sequence."""

    keypoints = build_normalized_keypoints(pose_data)
    keypoints, attention_mask = pad_or_sample_keypoints(keypoints, target_len=32)

    return {
        "input_kps": keypoints.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "name_batch": [sample_name],
        "pronunciation_batch": [""],
    }


def build_normalized_keypoints(pose_data):
    """Applies the same preprocessing used at evaluation time."""

    normalizer = build_pose_normalizer()
    normalized = normalizer(
        {
            "keypoints": np.asarray(pose_data["keypoints"], dtype=np.float32).copy(),
            "confidences": np.asarray(pose_data["confidences"], dtype=np.float32).copy(),
        }
    )["keypoints"]

    if not isinstance(normalized, torch.Tensor):
        normalized = torch.tensor(normalized, dtype=torch.float32)

    return normalized


def pad_or_sample_keypoints(keypoints, target_len):
    """Matches the temporal preprocessing used by the dataset collate function."""

    if keypoints.size(0) > target_len:
        indices = torch.linspace(0, keypoints.size(0) - 1, steps=target_len, device=keypoints.device).long()
        keypoints = keypoints[indices]
        attention_mask = torch.zeros(target_len, dtype=torch.bool, device=keypoints.device)
        return keypoints, attention_mask

    time_steps = keypoints.size(0)
    padded = torch.zeros((target_len, *keypoints.shape[1:]), dtype=keypoints.dtype, device=keypoints.device)
    padded[:time_steps] = keypoints

    if time_steps < target_len:
        padded[time_steps:] = keypoints[-1]

    attention_mask = torch.ones(target_len, dtype=torch.bool, device=keypoints.device)
    attention_mask[:time_steps] = False
    return padded, attention_mask


def load_model(checkpoint_path, device):
    """Loads a trained ProSign_generation checkpoint for inference."""

    model = ProSign_generation()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state_dict[normalized_key] = value

    result = model.load_state_dict(normalized_state_dict, strict=False)
    if result.missing_keys:
        print("Missing keys:")
        for key in result.missing_keys:
            print(key)
    if result.unexpected_keys:
        print("Unexpected keys:")
        for key in result.unexpected_keys:
            print(key)

    model.to(device)
    model.eval()
    return model


def run_inference(args):
    """Extracts pose, loads the model, and generates a pronunciation string."""

    device = torch.device(args.device)
    video_path = Path(args.video).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    pose_data, pose_path = load_pose_data(
        video_path=video_path,
        pose_dirname=args.pose_dirname,
        overwrite_pose=args.overwrite_pose,
    )
    src_input = build_src_input(pose_data, sample_name=video_path.stem)

    tokenizer = T5Tokenizer.from_pretrained(
        T5_MODEL_ID,
        local_files_only=use_local_t5_files_only(),
    )
    model = load_model(checkpoint_path, device=device)

    with torch.no_grad():
        outputs = model.generate(
            src_input,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=1,
        )

    pronunciation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Pose saved to: {pose_path}")
    print(f"Generated pronunciation: {pronunciation}")


if __name__ == "__main__":
    parser = get_args_parser()
    run_inference(parser.parse_args())

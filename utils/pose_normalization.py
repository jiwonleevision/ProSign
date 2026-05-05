import pickle
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


# keypoints의 크기 => (NUM_FRAME, NUM_KEYPOINTS(75), CHANNEL(X, Y, Z))

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class CenterNormalize:
    REFERENCE_PRESETS = {
        "shoulder_mediapipe_holistic": [11, 12, 23, 24]
    }

    def __init__(self, frame_level=False):
        self.REFERENCE_PRESETS = self.REFERENCE_PRESETS["shoulder_mediapipe_holistic"]
        self.frame_level = frame_level
    
    def __call__(self, data):
        keypoints = data["keypoints"].copy() # shape = TVC
        
        if self.frame_level:
            for idx in range(keypoints.shape[0]):
                center = self.calc_center_for_one_frame(keypoints[idx])
                keypoints[idx] -= center
        else:
            center = self.calc_center(keypoints)
            keypoints = keypoints - center

        data["keypoints"] = keypoints

        return data

    def calc_center_for_one_frame(self, keypoint):
        left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx = self.REFERENCE_PRESETS
        left_shoulder_keypoint, right_shoulder_keypoint, left_hip_keypoint, right_hip_keypoint = keypoint[left_shoulder_idx], keypoint[right_shoulder_idx], keypoint[left_hip_idx], keypoint[right_hip_idx]
        center = (left_shoulder_keypoint + right_shoulder_keypoint + left_hip_keypoint + right_hip_keypoint) / 4

        return center
    
    def calc_center(self, keypoints):
        transposed_keypoints = np.transpose(keypoints, (1, 0, 2)) # shape = VTC
        left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx = self.REFERENCE_PRESETS
        left_shoulder_keypoints = transposed_keypoints[left_shoulder_idx]
        right_shoulder_keypoints = transposed_keypoints[right_shoulder_idx]
        left_hip_keypoints = transposed_keypoints[left_hip_idx]
        right_hip_keypoints = transposed_keypoints[right_hip_idx]

        left_shoulder_keypoints = left_shoulder_keypoints.reshape(-1, left_shoulder_keypoints.shape[-1])
        right_shoulder_keypoints = right_shoulder_keypoints.reshape(-1, right_shoulder_keypoints.shape[-1])
        left_hip_keypoints = left_hip_keypoints.reshape(-1, left_hip_keypoints.shape[-1])
        right_hip_keypoints = right_hip_keypoints.reshape(-1, right_hip_keypoints.shape[-1])

        center = np.median((left_shoulder_keypoints + right_shoulder_keypoints + left_hip_keypoints + right_hip_keypoints) / 4, axis = 0)

        return center

class RemoveFace:
    def __call__(self, data):

        keypoints = data["keypoints"]

        if keypoints.shape[1] == 543:
            keypoints = keypoints[:, :75, :]

        data["keypoints"] = keypoints

        return data

class NumpyToTensor:
    def __call__(self, data):
        data["keypoints"] = torch.from_numpy(data["keypoints"]).float()
        return data
    
class TVC2CTV:
    def __call__(self, data):
        keypoints = data["keypoints"]
        data["keypoints"] = keypoints.permute(2, 0, 1)
        return data

class ConcatConfidence:
    def __call__(self,data):
        keypoints = data["keypoints"]
        confidences = data["confidences"]

        kps_conf = np.concatenate([keypoints, np.expand_dims(confidences, axis=-1)], axis=-1)

        return kps_conf

class ScaleNormalize:
    REFERENCE_PRESETS = {
        "shoulder_mediapipe_holistic": [11, 12, 23, 24]
    }

    def __init__(self, scale_factor=1, frame_level=False, eps=1e-6, clip_range=(0.5, 2.0)):
        self.REFERENCE_PRESETS = self.REFERENCE_PRESETS["shoulder_mediapipe_holistic"]
        self.scale_factor = scale_factor
        self.frame_level = frame_level
        self.eps = eps
        self.clip_range = clip_range
    
    def __call__(self, data):
        keypoints = data["keypoints"].copy()  # (T, V, C)
        
        if self.frame_level:
            for idx in range(keypoints.shape[0]):
                scale = self.get_scale_for_one_frame(keypoints[idx])
                keypoints[idx] *= scale
        else:
            scale = self.get_scale(keypoints)
            keypoints = keypoints * scale

        data["keypoints"] = keypoints
        return data

    def get_scale_for_one_frame(self, keypoint):
        left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx = self.REFERENCE_PRESETS

        ls = keypoint[left_shoulder_idx]
        rs = keypoint[right_shoulder_idx]
        lh = keypoint[left_hip_idx]
        rh = keypoint[right_hip_idx]

        shoulder_mean = (ls + rs) / 2
        hip_mean = (lh + rh) / 2
        left_mean = (ls + lh) / 2
        right_mean = (rs + rh) / 2

        torso_height = np.linalg.norm(shoulder_mean - hip_mean)
        torso_width = np.linalg.norm(left_mean - right_mean)

        torso_value = np.sqrt(torso_height**2 + torso_width**2)

        # 🔥 안정화 핵심 1: epsilon
        torso_value = max(torso_value, self.eps)

        scale = self.scale_factor / torso_value

        # 🔥 안정화 핵심 2: clipping
        scale = np.clip(scale, *self.clip_range)

        return scale
    
    def get_scale(self, keypoints):
        transposed_keypoints = np.transpose(keypoints, (1, 0, 2))  # (V, T, C)

        left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx = self.REFERENCE_PRESETS

        ls = transposed_keypoints[left_shoulder_idx].reshape(-1, 3)
        rs = transposed_keypoints[right_shoulder_idx].reshape(-1, 3)
        lh = transposed_keypoints[left_hip_idx].reshape(-1, 3)
        rh = transposed_keypoints[right_hip_idx].reshape(-1, 3)

        shoulder_mean = np.median((ls + rs) / 2, axis=0)
        hip_mean = np.median((lh + rh) / 2, axis=0)
        left_mean = np.median((ls + lh) / 2, axis=0)
        right_mean = np.median((rs + rh) / 2, axis=0)

        torso_height = np.linalg.norm(shoulder_mean - hip_mean)
        torso_width = np.linalg.norm(left_mean - right_mean)

        torso_value = np.sqrt(torso_height**2 + torso_width**2)

        # 🔥 안정화 핵심 1: epsilon
        torso_value = max(torso_value, self.eps)

        scale = self.scale_factor / torso_value

        # 🔥 안정화 핵심 2: clipping
        scale = np.clip(scale, *self.clip_range)

        return scale

class KeypointLinearInterpolation:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold

    def __call__(self, data):
        """
        sample["keypoints"]: (T, V, 3)
        """

        coords = data["keypoints"]
        conf = data["confidences"]

        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()

        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()

        # low confidence → NaN
        coords[conf < self.conf_threshold] = np.nan

        T, V, C = coords.shape
        time_idx = np.arange(T)

        for v in range(V):
            for c in range(C):

                temp = coords[:, v, c]
                valid = ~np.isnan(temp)

                # interpolation 가능한 경우만
                n_valid = valid.sum()
                
                if n_valid == 0:
                    coords[:, v, c] = 0
                    continue

                if n_valid == 1:
                    coords[:, v, c] = temp[valid][0]
                    continue

                coords[:, v, c] = np.interp(
                    time_idx,
                    time_idx[valid],
                    temp[valid]
                )


        data["keypoints"] = coords

        return data

class AddRelativeVelocity:
    def __init__(self, pad_mode="zero"):
        self.pad_mode = pad_mode  # "zero" or "repeat"

    def __call__(self, data):
        keypoints = data["keypoints"]  # (T, V, C)

        if isinstance(keypoints, torch.Tensor):
            keypoints_np = keypoints.cpu().numpy()
            is_tensor = True
        else:
            keypoints_np = keypoints
            is_tensor = False

        T = keypoints_np.shape[0]

        # 🔥 case 1: frame이 1개인 경우
        if T == 1:
            velocity = np.zeros_like(keypoints_np)

        # 🔥 case 2: 정상적인 경우
        else:
            velocity = keypoints_np[1:] - keypoints_np[:-1]  # (T-1, V, C)

            # padding (T 맞추기)
            if self.pad_mode == "zero":
                pad = np.zeros_like(velocity[0:1])
            elif self.pad_mode == "repeat":
                pad = velocity[0:1]
            else:
                raise ValueError("pad_mode must be 'zero' or 'repeat'")

            velocity = np.concatenate([pad, velocity], axis=0)  # (T, V, C)

        # 🔥 안전 체크 (디버깅용)
        if velocity.shape[0] != T:
            raise ValueError(f"Velocity length mismatch: {velocity.shape[0]} vs {T}")

        # concat → (T, V, 2C)
        keypoints_with_vel = np.concatenate([keypoints_np, velocity], axis=-1)

        # tensor로 복구
        if is_tensor:
            keypoints_with_vel = torch.from_numpy(keypoints_with_vel).float()

        data["keypoints"] = keypoints_with_vel

        return data
    

class TemporalDropout:
    def __init__(self, drop_ratio=0.2):
        self.drop_ratio = drop_ratio

    def __call__(self, data):
        keypoints = data["keypoints"]  # (T, V, C)
        T = keypoints.shape[0]

        keep_T = max(1, int(T * (1 - self.drop_ratio)))
        keep_indices = np.sort(
            np.random.choice(T, keep_T, replace=False)
        )

        keypoints = keypoints[keep_indices]

        data["keypoints"] = keypoints

        # confidences도 같이 처리
        if "confidences" in data:
            data["confidences"] = data["confidences"][keep_indices]

        return data
    
class TemporalResample:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, data):
        keypoints = data["keypoints"]  # (T, V, C)
        T, V, C = keypoints.shape

        scale = np.random.uniform(*self.scale_range)
        new_T = max(1, int(T * scale))

        old_time = np.linspace(0, 1, T)
        new_time = np.linspace(0, 1, new_T)

        new_keypoints = np.zeros((new_T, V, C))

        for v in range(V):
            for c in range(C):
                new_keypoints[:, v, c] = np.interp(
                    new_time,
                    old_time,
                    keypoints[:, v, c]
                )

        data["keypoints"] = new_keypoints

        # confidences도 같이 보간
        if "confidences" in data:
            conf = data["confidences"]
            new_conf = np.zeros((new_T, V))

            for v in range(V):
                new_conf[:, v] = np.interp(
                    new_time,
                    old_time,
                    conf[:, v]
                )

            data["confidences"] = new_conf

        return data
    
class JointMasking:
    def __init__(self, mask_ratio=0.1, mask_value=0):
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def __call__(self, data):
        keypoints = data["keypoints"]  # (T, V, C)
        T, V, C = keypoints.shape

        num_mask = int(V * self.mask_ratio)
        mask_joints = np.random.choice(V, num_mask, replace=False)

        keypoints[:, mask_joints, :] = self.mask_value

        data["keypoints"] = keypoints

        # confidence도 같이 0으로
        if "confidences" in data:
            data["confidences"][:, mask_joints] = 0

        return data

class RandomScale:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, data):
        keypoints = data["keypoints"]

        scale = np.random.uniform(*self.scale_range)
        keypoints = keypoints * scale

        data["keypoints"] = keypoints
        return data

class RandomTranslation:
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, data):
        keypoints = data["keypoints"]

        # (C,) shift vector
        shift = np.random.uniform(
            -self.translate_range,
            self.translate_range,
            size=(1, 1, keypoints.shape[-1])
        )

        keypoints = keypoints + shift

        data["keypoints"] = keypoints
        return data

class AddSpatialNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        keypoints = data["keypoints"]

        noise = np.random.normal(
            loc=0.0,
            scale=self.std,
            size=keypoints.shape
        )

        keypoints = keypoints + noise

        data["keypoints"] = keypoints
        return data

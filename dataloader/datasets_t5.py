import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import utils.pose_normalization as pn
import utils.video_normalization as vn

from transformers import T5Tokenizer

try:
    import cv2
except ImportError:
    cv2 = None


class SignCountryDataset(torch.utils.data.Dataset):
    """Dataset for ProSign pretraining and generation experiments."""

    DATASET_PATH = Path(os.environ.get("PROSIGN_DATASET_PATH", "path/to/dataset"))
    TEXT_FEATURE_DIRNAME = os.environ.get("PROSIGN_TEXT_FEATURE_DIRNAME", "t5_large_feat")
    KEYPOINTS_DIRNAME = os.environ.get("PROSIGN_KEYPOINTS_DIRNAME", "keypoints")
    VIDEO_DIRNAME = os.environ.get("PROSIGN_VIDEO_DIRNAME", "video_crop")
    METADATA_DIRNAME = os.environ.get("PROSIGN_METADATA_DIRNAME", "metadata")
    VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}

    def __init__(
        self,
        modality,
        split,
        include_text_feat=False,
        country=None,
        zero_shot=None,
        tokenizer_name="google-t5/t5-large",
        resize=(224, 224),
        fps=16,
    ):
        super().__init__()

        if modality not in {"keypoints", "video"}:
            raise ValueError(f"Unsupported modality: {modality}")

        if not self.DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.DATASET_PATH}")

        self.modality = modality
        self.split = split
        self.include_text_feat = include_text_feat
        self.country = country
        self.zero_shot = zero_shot
        self.resize = resize
        self.fps = fps
        self.country_list = self._resolve_countries(country)
        self.tokenizer = None if include_text_feat else T5Tokenizer.from_pretrained(tokenizer_name)
        self.normalizer = self._build_normalizer(split)

        self.data = self._build_dataframe()

        if self.modality == "keypoints":
            self._load_keypoints()

        if include_text_feat:
            self._attach_text_feature_paths()

        self.data = self._select_split(split, zero_shot).reset_index(drop=True)

    def _resolve_countries(self, country):
        available_countries = []

        for path in self.DATASET_PATH.iterdir():
            if not path.is_dir():
                continue

            modality_root = path / self._modality_dirname()
            if modality_root.exists():
                available_countries.append(path.name)

        if country is None:
            return available_countries
        if isinstance(country, list):
            return country
        return [country]

    def _modality_dirname(self):
        if self.modality == "keypoints":
            return self.KEYPOINTS_DIRNAME
        return self.VIDEO_DIRNAME

    def _build_normalizer(self, split):
        if self.modality == "keypoints":
            if split == "train":
                return pn.Compose(
                    [
                        pn.RemoveFace(),
                        pn.KeypointLinearInterpolation(),
                        pn.ScaleNormalize(scale_factor=1, frame_level=False),
                        pn.CenterNormalize(frame_level=False),
                        pn.RandomScale(scale_range=(0.95, 1.05)),
                        pn.RandomTranslation(translate_range=0.05),
                        pn.AddSpatialNoise(std=0.01),
                        pn.AddRelativeVelocity(),
                        pn.TemporalDropout(drop_ratio=0.2),
                        pn.TemporalResample(scale_range=(0.8, 1.2)),
                        pn.JointMasking(mask_ratio=0.1),
                        pn.NumpyToTensor(),
                    ]
                )

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

        if split == "train":
            return vn.Compose(
                [
                    vn.NumpyToTensor(),
                    vn.THWC2TCHW(),
                    vn.SpatialAugmentation(15, 5, (0.9, 1.1)),
                    vn.ColorJitterAugmentation(),
                    vn.ImageNetNormalize(),
                    vn.Resize(self.resize),
                    vn.FPSNormalize(self.fps),
                ]
            )

        return vn.Compose(
            [
                vn.NumpyToTensor(),
                vn.THWC2TCHW(),
                vn.ImageNetNormalize(),
                vn.Resize(self.resize),
                vn.FPSNormalize(self.fps),
            ]
        )

    def _build_dataframe(self):
        rows = []

        for country_name in self.country_list:
            country_path = self.DATASET_PATH / country_name
            sample_root = country_path / self._modality_dirname()
            metadata_df = self._load_metadata(country_path)

            for sample_path in self._iter_sample_files(sample_root):
                sample_meta = self._match_metadata_row(metadata_df, sample_path)
                if sample_meta is None:
                    continue

                rows.append(
                    {
                        "origin_no": sample_meta["origin_no"],
                        "country": sample_meta["country"],
                        "gloss": sample_meta["gloss"],
                        "pronunciation": sample_meta["pronunciation"],
                        f"{self.modality}_path": sample_path,
                    }
                )

        return pd.DataFrame(rows)

    def _load_metadata(self, country_path):
        metadata_root = country_path / self.METADATA_DIRNAME
        metadata_files = list(metadata_root.glob("*.csv"))

        if len(metadata_files) != 1:
            raise RuntimeError("Each country must provide exactly one metadata CSV file.")

        metadata_df = pd.read_csv(metadata_files[0], encoding="utf-8-sig").copy()
        metadata_df["origin_no"] = metadata_df["origin_no"].astype(str)
        metadata_df["origin_no_06d"] = metadata_df["origin_no"].astype(int).map("{:06d}".format)
        metadata_df["country"] = metadata_df.get("country", pd.Series(country_path.name, index=metadata_df.index))
        metadata_df["pronunciation"] = metadata_df.get(
            "pronunciation_reg",
            metadata_df.get("pronunciation", pd.Series("", index=metadata_df.index)),
        )
        return metadata_df

    def _iter_sample_files(self, root):
        if not root.exists():
            return

        for path in sorted(root.rglob("*")):
            if path.is_file() and self._is_supported_sample_file(path):
                yield path

    def _is_supported_sample_file(self, path):
        if self.modality == "keypoints":
            return path.suffix.lower() == ".pkl"
        return path.suffix.lower() in self.VIDEO_EXTENSIONS

    def _match_metadata_row(self, metadata_df, sample_path):
        for candidate in self._origin_candidates(sample_path):
            matched_df = metadata_df.loc[metadata_df["origin_no_06d"] == candidate].head(1)
            if not matched_df.empty:
                row = matched_df.iloc[0]
                return {
                    "origin_no": row["origin_no"],
                    "country": row["country"],
                    "gloss": row.get("gloss", ""),
                    "pronunciation": row["pronunciation"],
                }

        return None

    def _origin_candidates(self, sample_path):
        tokens = [
            sample_path.parent.name,
            sample_path.stem,
            sample_path.name,
        ]

        candidates = []
        seen = set()

        for token in tokens:
            self._append_origin_candidate(candidates, seen, token)
            for match in re.findall(r"\d+", token):
                self._append_origin_candidate(candidates, seen, match)

        return candidates

    @staticmethod
    def _append_origin_candidate(candidates, seen, token):
        if not token:
            return

        token = str(token).strip()
        if not token:
            return

        if token.isdigit():
            normalized = f"{int(token):06d}"
            if normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
            return

        if token not in seen:
            candidates.append(token)
            seen.add(token)

    def _load_keypoints(self):
        def load_pkl(path):
            with open(path, "rb") as file:
                return pickle.load(file)

        self.data["keypoints"] = self.data["keypoints_path"].apply(load_pkl)

    def _attach_text_feature_paths(self):
        def build_path(row):
            origin_no = int(row["origin_no"])
            return str(self.DATASET_PATH / row["country"] / self.TEXT_FEATURE_DIRNAME / f"{origin_no:06d}.embed")

        self.data["text_feat_path"] = self.data.apply(build_path, axis=1)

    def _select_split(self, split, zero_shot):
        if zero_shot is None:
            train_idx, val_idx = self._stratified_gloss_split()
            split_map = {"train": train_idx, "val": val_idx}
        elif zero_shot == "gs":
            train_idx, val_idx, test_idx = self._zero_shot_gloss_split()
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        elif zero_shot == "gzsl":
            train_idx, val_idx, test_idx = self._gzsl_split()
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        else:
            raise ValueError(f"Unsupported zero-shot setting: {zero_shot}")

        if split in split_map:
            return self.data.loc[split_map[split]]
        return self.data

    def _stratified_gloss_split(self, val_ratio=0.05, seed=42):
        np.random.seed(seed)
        train_indices = []
        val_indices = []

        for _, group in self.data.groupby("origin_no"):
            indices = group.index.tolist()

            if len(indices) == 1:
                train_indices.append(indices[0])
                continue

            if len(indices) == 2:
                train_indices.append(indices[0])
                val_indices.append(indices[1])
                continue

            np.random.shuffle(indices)
            train_indices.append(indices[0])
            remaining = indices[1:]
            val_size = int(len(remaining) * val_ratio)
            val_indices.extend(remaining[:val_size])
            train_indices.extend(remaining[val_size:])

        return train_indices, val_indices

    def _zero_shot_gloss_split(self, val_ratio=0.1, test_ratio=0.1, seed=42):
        np.random.seed(seed)
        gloss_list = self.data["origin_no"].unique().tolist()
        np.random.shuffle(gloss_list)

        val_size = int(len(gloss_list) * val_ratio)
        test_size = int(len(gloss_list) * test_ratio)

        val_gloss = gloss_list[:val_size]
        test_gloss = gloss_list[val_size:val_size + test_size]
        train_gloss = gloss_list[val_size + test_size:]

        train_idx = self.data[self.data["origin_no"].isin(train_gloss)].index.tolist()
        val_idx = self.data[self.data["origin_no"].isin(val_gloss)].index.tolist()
        test_idx = self.data[self.data["origin_no"].isin(test_gloss)].index.tolist()
        return train_idx, val_idx, test_idx

    def _gzsl_split(self, val_ratio=0.1, unseen_ratio=0.1, seed=42):
        np.random.seed(seed)
        gloss_list = self.data["origin_no"].unique().tolist()
        np.random.shuffle(gloss_list)

        unseen_size = int(len(gloss_list) * unseen_ratio)
        unseen_gloss = gloss_list[:unseen_size]
        seen_gloss = gloss_list[unseen_size:]

        seen_df = self.data[self.data["origin_no"].isin(seen_gloss)]
        unseen_df = self.data[self.data["origin_no"].isin(unseen_gloss)]

        train_idx = []
        val_idx = []
        test_seen_idx = []

        for _, group in seen_df.groupby("origin_no"):
            indices = group.index.tolist()
            np.random.shuffle(indices)
            val_size = max(1, int(len(indices) * val_ratio))
            test_size = max(1, int(len(indices) * val_ratio))

            val_idx.extend(indices[:val_size])
            test_seen_idx.extend(indices[val_size:val_size + test_size])
            train_idx.extend(indices[val_size + test_size:])

        test_idx = test_seen_idx + unseen_df.index.tolist()
        return train_idx, val_idx, test_idx

    def __len__(self):
        return len(self.data)

    def __len_glosses__(self):
        return self.data.groupby("country")["origin_no"].nunique().sum()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        if self.modality == "keypoints":
            sample = self._build_keypoints_sample(row)
        else:
            sample = self._build_video_sample(row)

        if self.include_text_feat:
            text_feat = torch.load(row["text_feat_path"], weights_only=False)
            if isinstance(text_feat, dict):
                if "t5_large_feat" in text_feat:
                    text_feat = text_feat["t5_large_feat"]
                elif "text_feat" in text_feat:
                    text_feat = text_feat["text_feat"]
                elif "embed" in text_feat:
                    text_feat = text_feat["embed"]
                else:
                    raise KeyError(f"Unknown text feature keys: {list(text_feat.keys())}")

            if not isinstance(text_feat, torch.Tensor):
                text_feat = torch.tensor(text_feat, dtype=torch.float32)
            sample["text_feat"] = text_feat.float()

        return sample

    def _build_keypoints_sample(self, row):
        keypoints_data = self._clone_sample_payload(row["keypoints"])
        if not isinstance(keypoints_data, dict):
            keypoints_data = {"keypoints": keypoints_data}

        keypoints = self.normalizer(keypoints_data)["keypoints"]
        return {
            "keypoints": keypoints,
            "name": self._path_to_name(row["keypoints_path"]),
            "pronunciation": row["pronunciation"],
        }

    def _build_video_sample(self, row):
        frames, original_fps = self._load_video_frames(row["video_path"])
        frames = self.normalizer({"frames": frames, "ofps": original_fps})["frames"]
        return {
            "frames": frames,
            "name": self._path_to_name(row["video_path"]),
            "pronunciation": row["pronunciation"],
        }

    def _load_video_frames(self, path):
        if cv2 is None:
            raise ImportError("opencv-python is required to use the video modality.")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {path}")

        frames = []

        try:
            original_fps = capture.get(cv2.CAP_PROP_FPS)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"Video file contains no readable frames: {path}")

        if not original_fps or original_fps <= 0:
            original_fps = self.fps

        return np.stack(frames), original_fps

    @staticmethod
    def _clone_sample_payload(payload):
        if isinstance(payload, dict):
            cloned = {}
            for key, value in payload.items():
                if isinstance(value, np.ndarray):
                    cloned[key] = value.copy()
                elif torch.is_tensor(value):
                    cloned[key] = value.clone()
                else:
                    cloned[key] = value
            return cloned

        if isinstance(payload, np.ndarray):
            return payload.copy()

        if torch.is_tensor(payload):
            return payload.clone()

        return payload

    def collate_fn(self, batch):
        if self.modality == "keypoints":
            return self._collate_keypoints_batch(batch)
        return self._collate_video_batch(batch)

    def _collate_keypoints_batch(self, batch):
        name_batch = []
        pronunciation_batch = []
        keypoints_list = []
        mask_list = []
        text_feat_list = []
        target_len = 32

        for sample in batch:
            name_batch.append(sample["name"])
            pronunciation_batch.append(sample["pronunciation"])

            keypoints = sample["keypoints"]
            if not isinstance(keypoints, torch.Tensor):
                keypoints = torch.tensor(keypoints, dtype=torch.float32)

            if keypoints.size(0) > target_len:
                indices = torch.linspace(0, keypoints.size(0) - 1, steps=target_len, device=keypoints.device).long()
                keypoints = keypoints[indices]
                mask = torch.zeros(target_len, dtype=torch.bool, device=keypoints.device)
            else:
                keypoints, mask = self._pad_sequence(keypoints, target_len)

            keypoints_list.append(keypoints)
            mask_list.append(mask)

            if self.include_text_feat:
                text_feat_list.append(sample["text_feat"].squeeze())

        src_input = {
            "input_kps": torch.stack(keypoints_list, dim=0),
            "attention_mask": torch.stack(mask_list, dim=0),
            "name_batch": name_batch,
            "pronunciation_batch": pronunciation_batch,
        }

        return src_input, self._build_target_batch(pronunciation_batch, text_feat_list)

    def _collate_video_batch(self, batch):
        name_batch = []
        pronunciation_batch = []
        frame_list = []
        mask_list = []
        text_feat_list = []
        target_len = 32

        for sample in batch:
            name_batch.append(sample["name"])
            pronunciation_batch.append(sample["pronunciation"])

            frames = sample["frames"]
            if not isinstance(frames, torch.Tensor):
                frames = torch.tensor(frames, dtype=torch.float32)

            if frames.size(0) > target_len:
                indices = torch.linspace(0, frames.size(0) - 1, steps=target_len, device=frames.device).long()
                frames = frames[indices]
                mask = torch.zeros(target_len, dtype=torch.bool, device=frames.device)
            else:
                frames, mask = self._pad_sequence(frames, target_len)

            frame_list.append(frames)
            mask_list.append(mask)

            if self.include_text_feat:
                text_feat_list.append(sample["text_feat"].squeeze())

        src_input = {
            "input_img": torch.stack(frame_list, dim=0),
            "attention_mask": torch.stack(mask_list, dim=0),
            "name_batch": name_batch,
            "pronunciation_batch": pronunciation_batch,
        }

        return src_input, self._build_target_batch(pronunciation_batch, text_feat_list)

    def _build_target_batch(self, pronunciation_batch, text_feat_list):
        if self.include_text_feat:
            return torch.stack(text_feat_list, dim=0)

        return self.tokenizer(
            pronunciation_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

    @staticmethod
    def _pad_sequence(sequence, target_len):
        time_steps = sequence.size(0)
        padded = torch.zeros((target_len, *sequence.shape[1:]), dtype=sequence.dtype, device=sequence.device)
        padded[:time_steps] = sequence

        if time_steps < target_len:
            padded[time_steps:] = sequence[-1]

        mask = torch.ones(target_len, dtype=torch.bool, device=sequence.device)
        mask[:time_steps] = False
        return padded, mask

    @staticmethod
    def _path_to_name(path):
        if path.parent.name == path.stem:
            return path.stem
        return f"{path.parent.name}_{path.stem}"

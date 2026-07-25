# ProSign

Multilingual sign-to-pronunciation training and evaluation codebase built around the ProSign pipeline.

This repository focuses on:

- keypoint-based pretraining with paired text features
- sign-to-pronunciation generation with T5
- evaluation and single-video inference from a trained checkpoint

The README is intentionally structured in a dataset-card style: quick to scan, explicit about inputs, and ready to run.

## Overview

`ProSign` is organized as a two-stage workflow:

1. `train_prosign_pretrain.py`
   Learns a cross-modal representation between sign keypoints and text features.
2. `train_prosign_generation.py`
   Fine-tunes a generation model that predicts pronunciations from sign keypoints.
3. `infer_prosign_generation.py`
   Runs inference on a single video by extracting MediaPipe keypoints and decoding with the trained generator.

Core directories:

- `dataloader/`: dataset loading and split construction
- `models/`: pretraining and generation model definitions
- `utils/`: normalization, metrics, distributed helpers

## Expected Dataset Layout

The loader reads the dataset root from `PROSIGN_DATASET_PATH`.

```text
${PROSIGN_DATASET_PATH}/
|-- Country A (Language A)/
|   |-- metadata/
|   |   `-- *.csv
|   |-- keypoints/
|   |   `-- **/*.pkl
|   |-- t5_large_feat/
|   |   `-- *.embed
|   `-- video_crop/
|       `-- **/*.mp4
|-- Country B (Language B)/
|   `-- ...
`-- ...
```

Metadata expectations inferred from the code:

- one CSV file per country under `metadata/`
- required identifier column: `origin_no`
- text columns used by the pipeline: `gloss`, `pronunciation`, or `pronunciation_reg`
- optional `country` column; if missing, the folder name is used

Environment variables supported by the loader:

- `PROSIGN_DATASET_PATH`
- `PROSIGN_TEXT_FEATURE_DIRNAME` default: `t5_large_feat`
- `PROSIGN_KEYPOINTS_DIRNAME` default: `keypoints`
- `PROSIGN_VIDEO_DIRNAME` default: `video_crop`
- `PROSIGN_METADATA_DIRNAME` default: `metadata`

## Setup

Install PyTorch separately for your CUDA environment first, then install the rest:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Windows PowerShell example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Set the dataset root before running experiments:

```bash
export PROSIGN_DATASET_PATH=/path/to/prosign_dataset
export TOKENIZERS_PARALLELISM=false
```

```powershell
$env:PROSIGN_DATASET_PATH="C:\path\to\prosign_dataset"
$env:TOKENIZERS_PARALLELISM="false"
```

## Model Download

There are two kinds of weights involved in this repository:

1. Base language model weights
   `train_prosign_generation.py` and `infer_prosign_generation.py` use `google-t5/t5-large` as the text backbone. These weights are downloaded automatically from Hugging Face on first run:

   - Hugging Face model page: [google-t5/t5-large](https://huggingface.co/google-t5/t5-large)

2. ProSign task checkpoints
   This repository expects task-specific checkpoints such as `best_checkpoint.pth`, but they are not bundled in the codebase. In practice, you should use one of the following:

   - a checkpoint you trained yourself with `train_prosign_pretrain.py` or `train_prosign_generation.py`
   - Google Drive folder for inference: [ProSign inference checkpoint](https://drive.google.com/drive/folders/17H-hO16LYTNUEobXgxIe-W-3Rb7O8D4w?usp=sharing)

After downloading, place the checkpoint somewhere like:

```text
checkpoints/
`-- prosign_inference_best_checkpoint.pth
```

Then use it in inference like this:

```bash
python infer_prosign_generation.py \
  --checkpoint checkpoints/prosign_inference_best_checkpoint.pth \
  --video /path/to/sample.mp4
```

Recommended local layout:

```text
outputs/
|-- prosign_pretrain_china_to_usa/
|   `-- best_checkpoint.pth
`-- prosign_generation_china_to_usa/
    `-- best_checkpoint.pth
```

If you already have a downloaded checkpoint, point the scripts to it with `--finetune`, `--resume`, or `--checkpoint`.

## Quickstart

Single-GPU or single-process runs can be launched with plain `python`.
Multi-GPU runs should use `torchrun`.

Example country pair used below:

- train country: `China (Chinese)`
- test country: `USA (English)`

Replace them with the actual country names that exist in your dataset folders.

## Train

### 1. Pretrain

```bash
python train_prosign_pretrain.py \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --batch-size 16 \
  --epochs 80 \
  --output_dir outputs/prosign_pretrain_china_to_usa
```

Multi-GPU example:

```bash
torchrun --nproc_per_node=4 train_prosign_pretrain.py \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --batch-size 16 \
  --epochs 80 \
  --output_dir outputs/prosign_pretrain_china_to_usa
```

### 2. Train the generator

Use the best pretraining checkpoint as `--finetune`.

```bash
python train_prosign_generation.py \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --batch-size 16 \
  --epochs 80 \
  --finetune outputs/prosign_pretrain_china_to_usa/best_checkpoint.pth \
  --output_dir outputs/prosign_generation_china_to_usa
```

Multi-GPU example:

```bash
torchrun --nproc_per_node=4 train_prosign_generation.py \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --batch-size 16 \
  --epochs 80 \
  --finetune outputs/prosign_pretrain_china_to_usa/best_checkpoint.pth \
  --output_dir outputs/prosign_generation_china_to_usa
```

### Optional zero-shot settings

Supported values from the dataset loader:

- `--zero_shot gs`
- `--zero_shot gzsl`

Example:

```bash
python train_prosign_generation.py \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --zero_shot gzsl \
  --finetune outputs/prosign_pretrain_china_to_usa/best_checkpoint.pth \
  --output_dir outputs/prosign_generation_gzsl
```

## Test

### Pretraining checkpoint evaluation

```bash
python train_prosign_pretrain.py \
  --eval \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --resume outputs/prosign_pretrain_china_to_usa/best_checkpoint.pth \
  --output_dir outputs/prosign_pretrain_eval
```

This reports retrieval-style metrics such as `Top1`, `Top5`, and `Top10`.

### Generation checkpoint evaluation

```bash
python train_prosign_generation.py \
  --eval \
  --country_train "China (Chinese)" \
  --country_test "USA (English)" \
  --resume outputs/prosign_generation_china_to_usa/best_checkpoint.pth \
  --output_dir outputs/prosign_generation_eval
```

This reports generation metrics such as `BLEU-4`, `ROUGE-1`, `ROUGE-2`, and `ROUGE-L`.

## Reproducing Paper Results

The ProSign benchmark trains on 35 country-language entries and evaluates on 7 unseen countries. `--country_train` accepts multiple entries, for example `--country_train "China (Chinese)" "USA (English)"`, while `--country_test` takes a single country per run. Folder names must match the country directories under `PROSIGN_DATASET_PATH`. All commands below also require `--config`; a minimal example is provided at `configs/prosign.yaml`.

Define the benchmark split once:

```bash
TRAIN_COUNTRIES=(
  "USA (English)" "China (Chinese)" "Germany (German)" "Japan (Japanese)"
  "India (Hindi)" "UK (English)" "France (French)" "Italy (Italian)"
  "Russia (Russian)" "Brazil (Portuguese)" "Spain (Spanish)" "Mexico (Spanish)"
  "South Korea (Korean)" "Turkey (Turkish)" "Poland (Polish)" "Sweden (Swedish)"
  "Argentina (Spanish)" "Vietnam (Vietnamese)" "Denmark (Danish)" "Finland (Finnish)"
  "Portugal (Portuguese)" "Romania (Romanian)" "Czech Republic (Czech)" "Pakistan (Urdu)"
  "Greece (Greek)" "Ukraine (Ukrainian)" "Bulgaria (Bulgarian)" "Chile (Spanish)"
  "Slovakia (Slovak)" "Kenya (English)" "Belarus (Russian)" "Belarus (Belarusian)"
  "Serbia (Serbian)" "Lithuania (Lithuanian)" "International (International Signs)"
)
TEST_COUNTRIES=(
  "Croatia (Croatian)" "Cuba (Spanish)" "Cyprus (Greek)" "Estonia (Estonian)"
  "Iceland (Icelandic)" "Latvia (Latvian)" "Syria (Arabic)"
)
```

### 1. Alignment pretraining (paper setting: 50 epochs, batch size 24)

During training, `--country_test` specifies the country evaluated at each epoch, and the best checkpoint is tracked by its Top1. The paper runs used `Croatia (Croatian)`.

```bash
torchrun --nproc_per_node=8 train_prosign_pretrain.py \
  --country_train "${TRAIN_COUNTRIES[@]}" \
  --country_test "Croatia (Croatian)" \
  --batch-size 24 \
  --epochs 50 \
  --config configs/prosign.yaml \
  --output_dir outputs/prosign_pretrain_benchmark
```

### 2. Zero-shot recognition evaluation (Table 2)

Evaluate each unseen country in turn:

```bash
for TC in "${TEST_COUNTRIES[@]}"; do
  python train_prosign_pretrain.py \
    --eval \
    --country_train "${TRAIN_COUNTRIES[@]}" \
    --country_test "$TC" \
    --resume outputs/prosign_pretrain_benchmark/best_checkpoint.pth \
    --config configs/prosign.yaml \
    --output_dir outputs/prosign_pretrain_benchmark_eval
done
```

Each run reports `Top1`, `Top5`, and `Top10` for that country and appends them to `<country>_test_log.txt` under the output directory.

### 3. Pronunciation generation training (paper setting: 30 epochs, batch size 2)

```bash
torchrun --nproc_per_node=8 train_prosign_generation.py \
  --country_train "${TRAIN_COUNTRIES[@]}" \
  --country_test "Croatia (Croatian)" \
  --batch-size 2 \
  --epochs 30 \
  --finetune outputs/prosign_pretrain_benchmark/best_checkpoint.pth \
  --config configs/prosign.yaml \
  --output_dir outputs/prosign_generation_benchmark
```

### 4. Pronunciation generation evaluation (Table 3)

```bash
for TC in "${TEST_COUNTRIES[@]}"; do
  python train_prosign_generation.py \
    --eval \
    --country_train "${TRAIN_COUNTRIES[@]}" \
    --country_test "$TC" \
    --resume outputs/prosign_generation_benchmark/best_checkpoint.pth \
    --config configs/prosign.yaml \
    --output_dir outputs/prosign_generation_benchmark_eval
done
```

Each run reports `BLEU-4` and `ROUGE-1/2/L` for that country.

## Inference

Run inference on a single video file with a trained generation checkpoint:

```bash
python infer_prosign_generation.py \
  --checkpoint outputs/prosign_generation_china_to_usa/best_checkpoint.pth \
  --video /path/to/sample.mp4
```

Useful flags:

- `--device cpu`
- `--pose-dirname pose`
- `--overwrite-pose`
- `--max-new-tokens 150`
- `--num-beams 2`

Example:

```bash
python infer_prosign_generation.py \
  --checkpoint outputs/prosign_generation_china_to_usa/best_checkpoint.pth \
  --video demo/sample.mp4 \
  --pose-dirname pose_cache \
  --num-beams 4
```

The script will:

1. extract pose keypoints with MediaPipe Holistic
2. cache them as `pose/<video_stem>.pkl`
3. generate a pronunciation string with the trained T5-based decoder

## Outputs

Typical artifacts written under `--output_dir`:

- `checkpoint.pth`
- `best_checkpoint.pth`
- `log.txt`
- generation previews such as `epoch_XX_test_generate.txt`

## Notes

- `train_prosign_generation.py` downloads `google-t5/t5-large` on first use.
- `infer_prosign_generation.py` requires `opencv-python` and `mediapipe`.
- If you run distributed training, the scripts expect `torchrun`-style environment variables.
- If your dataset uses different folder names, set the corresponding `PROSIGN_*` environment variables before launch.

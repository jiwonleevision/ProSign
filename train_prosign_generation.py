import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import yaml

import hpargparse
import utils.utils as utils

from dataloader.datasets_t5 import SignCountryDataset
from hpman.m import _
from loguru import logger
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from timm.optim import create_optimizer
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler as scheduler
from transformers import T5ForConditionalGeneration, T5Tokenizer

from models.model_prosign_generation import ProSign_generation, T5_MODEL_ID


def sanitize_filename(value):
    """Sanitizes a string so it can be used safely as a filename."""

    sanitized = re.sub(r'[\\/:*?"<>|]+', "_", str(value)).strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized or "unknown"



def distributed_barrier():
    """Synchronizes all ranks when distributed training is active."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def prepare_t5_assets_for_distributed(model_id):
    """Downloads T5 assets on rank 0 and switches other ranks to local-cache mode."""

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        os.environ.pop("MMSLT_T5_LOCAL_FILES_ONLY", None)
        return

    if utils.is_main_process():
        T5Tokenizer.from_pretrained(model_id)
        T5ForConditionalGeneration.from_pretrained(model_id)

    distributed_barrier()
    os.environ["MMSLT_T5_LOCAL_FILES_ONLY"] = "1"


def get_args_parser():
    parser = argparse.ArgumentParser("ProSign_generation", add_help=False)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--country_train", default=["Country A (Language A)"], nargs="+", type=str)
    parser.add_argument("--country_test", default="Country B (Language B)", type=str)
    parser.add_argument("--zero_shot", default="", type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--finetune", default="")
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt-eps", default=1.0e-09, type=float, metavar="EPSILON")
    parser.add_argument("--opt-betas", default=[0.9, 0.98], type=float, nargs="+", metavar="BETA")
    parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER")
    parser.add_argument("--lr", type=float, default=1.0e-3, metavar="LR")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV")
    parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=1.0e-08, metavar="LR")
    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N")
    parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N")
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist-eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--config", type=str, default="path/to/config.yaml")
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)
    return parser


def build_dataloader(dataset, args, shuffle):
    """Builds a dataloader and matching sampler for distributed or single-process runs."""

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=args.pin_mem,
    )
    return sampler, dataloader


@record
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if args.zero_shot == "":
        args.zero_shot = None

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print("Using ProSign_generation module: prosign_generation")
    print(f"Using T5 tokenizer: {T5_MODEL_ID}")
    prepare_t5_assets_for_distributed(T5_MODEL_ID)

    tokenizer = T5Tokenizer.from_pretrained(
        T5_MODEL_ID,
        local_files_only=os.environ.get("MMSLT_T5_LOCAL_FILES_ONLY", "0") == "1",
    )

    train_data = SignCountryDataset(
        modality="keypoints",
        split="train",
        country=args.country_train,
        zero_shot=args.zero_shot,
        tokenizer_name=T5_MODEL_ID,
    )
    val_data = SignCountryDataset(
        modality="keypoints",
        split="val",
        country=args.country_train,
        zero_shot=args.zero_shot,
        tokenizer_name=T5_MODEL_ID,
    )
    test_data = SignCountryDataset(
        modality="keypoints",
        split=None,
        country=args.country_test,
        tokenizer_name=T5_MODEL_ID,
    )

    print(f'The number of samples in "TRAIN" data : {len(train_data)}')
    print(f'The number of glosses in "TRAIN" data : {train_data.__len_glosses__()}')
    print(f'The number of samples in "VALIDATION" data : {len(val_data)}')
    print(f'The number of glosses in "VALIDATION" data : {val_data.__len_glosses__()}')
    print(f'The number of samples in "TEST" data : {len(test_data)}')
    print(f'The number of glosses in "TEST" data : {test_data.__len_glosses__()}')

    train_sampler, train_dataloader = build_dataloader(train_data, args, shuffle=True)
    _, val_dataloader = build_dataloader(val_data, args, shuffle=False)
    _, test_dataloader = build_dataloader(test_data, args, shuffle=False)

    model = ProSign_generation()
    model.to(device)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if key.startswith("model_image."):
                new_state_dict[key] = value
            elif key.startswith("module.model_image."):
                new_state_dict[key.replace("module.", "", 1)] = value
            elif key.startswith("backbone."):
                new_state_dict["model_image." + key[len("backbone."):]] = value

        result = model.load_state_dict(new_state_dict, strict=False)
        print("Loaded keys:", len(new_state_dict))
        print("Missing keys:\n", "\n".join(result.missing_keys))
        print("Unexpected keys:\n", "\n".join(result.unexpected_keys))

        for _, param in model.model_image.named_parameters():
            param.requires_grad = False

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    print(f"number of params: {utils.count_parameters_in_MB(model_without_ddp)}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=1e-8,
        T_max=args.epochs,
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.2,
    )
    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        if not args.eval and all(key in checkpoint for key in ["optimizer", "lr_scheduler", "epoch"]):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        if not args.resume:
            logger.warning("Please specify a checkpoint with --resume.")
        val_stats = evaluate(args, val_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=None, split="val")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=None, split="test")
        print(f'Validation loss: {val_stats["loss"]:.4f}')
        print(f'Test loss: {test_stats["loss"]:.4f} | BLEU-4 {test_stats["belu4"]:.2f} | ROUGE-1 {test_stats["rouge1"]:.2f} | ROUGE-2 {test_stats["rouge2"]:.2f} | ROUGE-L {test_stats["rougeL"]:.2f}')
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_bleu = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch)
        lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                output_dir / "checkpoint.pth",
            )

        val_stats = evaluate(args, val_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=epoch, split="val")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=epoch, split="test")

        print(f'Validation loss: {val_stats["loss"]:.4f}')
        print(f'Test loss: {test_stats["loss"]:.4f} | BLEU-4 {test_stats["belu4"]:.2f} | ROUGE-1 {test_stats["rouge1"]:.2f} | ROUGE-2 {test_stats["rouge2"]:.2f} | ROUGE-L {test_stats["rougeL"]:.2f}')

        if test_stats["belu4"] >= best_bleu:
            best_bleu = test_stats["belu4"]
            if args.output_dir and utils.is_main_process():
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    output_dir / "best_checkpoint.pth",
                )

        print(f"Best BLEU-4: {best_bleu:.2f}")
        if utils.is_main_process():
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training/train_loss": train_stats["loss"],
                    "validation/loss": val_stats["loss"],
                    "test/bleu4": test_stats["belu4"],
                    "test/best_bleu4": best_bleu,
                }
            )

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
            **{f"test_{key}": value for key, value in test_stats.items()},
            "epoch": epoch,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a", encoding="utf-8") as file:
                file.write(json.dumps(log_stats) + "\n")

    if args.output_dir:
        distributed_barrier()
        checkpoint = torch.load(output_dir / "best_checkpoint.pth", map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        val_stats = evaluate(args, val_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=args.epochs - 1, split="val")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=args.epochs - 1, split="test")
        print(f'Best validation loss: {val_stats["loss"]:.4f}')
        print(f'Best test loss: {test_stats["loss"]:.4f} | BLEU-4 {test_stats["belu4"]:.2f} | ROUGE-1 {test_stats["rouge1"]:.2f} | ROUGE-2 {test_stats["rouge2"]:.2f} | ROUGE-L {test_stats["rougeL"]:.2f}')

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")


def train_one_epoch(args, model, criterion, dataloader, optimizer, device, epoch):
    """Runs one epoch of ProSign_generation optimization."""

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"

    for src_input, tgt_input in metric_logger.log_every(dataloader, 10, header):
        logits = model(src_input, tgt_input)
        labels = tgt_input["input_ids"].reshape(-1)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.to(device, non_blocking=True))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if len(optimizer.param_groups) > 1:
            metric_logger.update(lr_decoder=round(float(optimizer.param_groups[1]["lr"]), 8))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def evaluate(args, dataloader, model, model_without_ddp, tokenizer, criterion, device, epoch=None, split="val"):
    """Computes loss and, for the test split, generation metrics."""

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    compute_generation_metrics = split == "test"
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True) if compute_generation_metrics else None

    with torch.no_grad():
        predictions = []
        references = []
        names = []

        for src_input, tgt_input in metric_logger.log_every(dataloader, 10, "Eval"):
            logits = model(src_input, tgt_input)
            labels = tgt_input["input_ids"].reshape(-1)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.to(device))
            metric_logger.update(loss=loss.item())

            if not compute_generation_metrics:
                continue

            outputs = model_without_ddp.generate(
                src_input,
                max_new_tokens=150,
                num_beams=2,
                num_return_sequences=1,
            )
            target_ids = tgt_input["input_ids"].to(device)

            for batch_index in range(target_ids.shape[0]):
                predictions.append(outputs[batch_index])
                references.append(target_ids[batch_index])
                names.append(src_input["name_batch"][batch_index])

    if not compute_generation_metrics:
        metric_logger.synchronize_between_processes()
        print(f'* loss {metric_logger.loss.global_avg:.3f}')
        return {key: meter.global_avg for key, meter in metric_logger.meters.items()}

    predictions = tokenizer.batch_decode(
        pad_sequence(predictions, batch_first=True, padding_value=tokenizer.pad_token_id),
        skip_special_tokens=True,
    )
    references = tokenizer.batch_decode(
        pad_sequence(references, batch_first=True, padding_value=tokenizer.pad_token_id),
        skip_special_tokens=True,
    )

    if utils.get_world_size() > 1:
        names = utils.all_gather_object_list(names)
        predictions = utils.all_gather_object_list(predictions)
        references = utils.all_gather_object_list(references)

        merged_samples = OrderedDict()
        for name, prediction, reference in zip(names, predictions, references):
            if name not in merged_samples:
                merged_samples[name] = {"prediction": prediction, "reference": reference}

        names = list(merged_samples.keys())
        predictions = [sample["prediction"] for sample in merged_samples.values()]
        references = [sample["reference"] for sample in merged_samples.values()]

    bleu4 = BLEU().corpus_score(predictions, [references]).score
    metric_logger.meters["belu4"].update(bleu4)

    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l = 0.0
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rouge_l += scores["rougeL"].fmeasure

    count = max(len(predictions), 1)
    metric_logger.meters["rouge1"].update(rouge1 / count * 100)
    metric_logger.meters["rouge2"].update(rouge2 / count * 100)
    metric_logger.meters["rougeL"].update(rouge_l / count * 100)
    metric_logger.synchronize_between_processes()

    print(
        "* BLEU-4 {bleu.global_avg:.3f} | R1 {r1.global_avg:.3f} | R2 {r2.global_avg:.3f} | RL {rl.global_avg:.3f} | loss {losses.global_avg:.3f}".format(
            bleu=metric_logger.belu4,
            r1=metric_logger.rouge1,
            r2=metric_logger.rouge2,
            rl=metric_logger.rougeL,
            losses=metric_logger.loss,
        )
    )

    if utils.is_main_process():
        preview_size = min(10, len(predictions))
        for index in range(preview_size):
            print(f"{names[index]}: {predictions[index]}")

        if args.output_dir:
            epoch_str = str(epoch) if epoch is not None else "final"
            if args.eval and split == "test":
                country_slug = sanitize_filename(args.country_test)
                output_path = Path(args.output_dir) / f"epoch_{epoch_str}_{split}_{country_slug}_generate.txt"
                save_names = names
                save_predictions = predictions
            else:
                output_path = Path(args.output_dir) / f"epoch_{epoch_str}_{split}_generate.txt"
                save_names = names[:preview_size]
                save_predictions = predictions[:preview_size]

            with output_path.open("w", encoding="utf-8") as file:
                for name, prediction in zip(save_names, save_predictions):
                    file.write(f"{name}: {prediction}\n")

    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("ProSign_generation", parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["WANDB_MODE"] = config["training"]["wandb"] if not args.eval else "disabled"
    if utils.is_main_process():
        wandb.init(project="", config=config)
        wandb.run.name = args.output_dir.split("/")[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)

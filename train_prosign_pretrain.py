import argparse
import datetime
import json
import math
import os
import sys
import time

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import wandb
import yaml

import hpargparse
import utils.utils as utils

from dataloader.datasets_t5 import SignCountryDataset
from hpman.m import _
from loguru import logger
from models.model_prosign_pretrain import ProSign_pretrain
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("ProSign_pretrain", add_help=False)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--country_train", default="Country A (Language A)", nargs="+", type=str)
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
    parser.add_argument("--weight-decay", type=float, default=0.05)
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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--config", type=str, default="path/to/config.yaml")
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--log_all", action="store_true")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--project", type=str, default="")
    return parser


def build_dataloader(dataset, args, shuffle, drop_last=False):
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
        drop_last=drop_last,
    )
    return sampler, dataloader


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if args.zero_shot == "":
        args.zero_shot = None

    tokenizer_name = "google-t5/t5-large"
    train_data = SignCountryDataset(
        modality="keypoints",
        split="train",
        include_text_feat=True,
        country=args.country_train,
        zero_shot=args.zero_shot,
        tokenizer_name=tokenizer_name,
    )
    val_data = SignCountryDataset(
        modality="keypoints",
        split="val",
        include_text_feat=True,
        country=args.country_train,
        zero_shot=args.zero_shot,
        tokenizer_name=tokenizer_name,
    )
    test_data = SignCountryDataset(
        modality="keypoints",
        split=None,
        include_text_feat=True,
        country=args.country_test,
        tokenizer_name=tokenizer_name,
    )

    print(f'The number of samples in "TRAIN" data : {len(train_data)}')
    print(f'The number of glosses in "TRAIN" data : {train_data.__len_glosses__()}')
    print(f'The number of samples in "VALIDATION" data : {len(val_data)}')
    print(f'The number of glosses in "VALIDATION" data : {val_data.__len_glosses__()}')
    print(f'The number of samples in "TEST" data : {len(test_data)}')
    print(f'The number of glosses in "TEST" data : {test_data.__len_glosses__()}')

    train_sampler, train_dataloader = build_dataloader(train_data, args, shuffle=True, drop_last=True)
    _, val_dataloader = build_dataloader(val_data, args, shuffle=False)
    _, test_dataloader = build_dataloader(test_data, args, shuffle=False)

    print("Using ProSign_pretrain module: prosign_pretrain")
    model = ProSign_pretrain()
    model.to(device)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        result = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:\n", "\n".join(result.missing_keys))
        print("Unexpected keys:\n", "\n".join(result.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    print(f"number of params: {utils.count_parameters_in_MB(model_without_ddp)}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_scaler = NativeScaler()
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
        val_stats = evaluate(args, val_dataloader, model, criterion, args.start_epoch)
        test_stats = evaluate(args, test_dataloader, model, criterion, args.start_epoch)
        print(f'Validation loss: {val_stats["loss"]:.3f} | Top1 {val_stats["top1"]:.3f} | Top5 {val_stats["top5"]:.3f} | Top10 {val_stats["top10"]:.3f}')
        print(f'Test loss: {test_stats["loss"]:.3f} | Top1 {test_stats["top1"]:.3f} | Top5 {test_stats["top5"]:.3f} | Top10 {test_stats["top10"]:.3f}')
        if args.output_dir and utils.is_main_process():
            with (output_dir / f"{args.country_test}_test_log.txt").open("a", encoding="utf-8") as file:
                file.write(json.dumps({f"test_{key}": value for key, value in test_stats.items()}) + "\n")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_top1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, epoch, loss_scaler)
        lr_scheduler.step(epoch)

        if args.output_dir:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                output_dir / "checkpoint.pth",
            )

        val_stats = evaluate(args, val_dataloader, model, criterion, epoch)
        test_stats = evaluate(args, test_dataloader, model, criterion, epoch)

        if test_stats["top1"] >= best_top1:
            best_top1 = test_stats["top1"]
            if args.output_dir:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    output_dir / "best_checkpoint.pth",
                )

        print(f'* Test Top1 {test_stats["top1"]:.3f} | Best Top1 {best_top1:.3f}')

        if utils.is_main_process() and args.run:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "training/train_loss": train_stats["loss"],
                    "validation/loss": val_stats["loss"],
                    "test/best_top1": best_top1,
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        checkpoint = torch.load(output_dir / "best_checkpoint.pth", map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        val_stats = evaluate(args, val_dataloader, model, criterion, args.epochs - 1)
        test_stats = evaluate(args, test_dataloader, model, criterion, args.epochs - 1)
        print(f'Best validation loss: {val_stats["loss"]:.3f} | Top1 {val_stats["top1"]:.3f}')
        print(f'Best test loss: {test_stats["loss"]:.3f} | Top1 {test_stats["top1"]:.3f}')

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")


def train_one_epoch(args, model, criterion, dataloader, optimizer, epoch, loss_scaler):
    """Runs one epoch of ProSign_pretrain optimization."""

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"

    for src_input, tgt_input in metric_logger.log_every(dataloader, 10, header):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            norm_text, norm_images = model(src_input, tgt_input)
            text_all = utils.gather_features_with_grad(norm_text)
            image_all = utils.gather_features_with_grad(norm_images)
            logit_scale = getattr(model, "module", model).logit_scale.exp().clamp(max=30)
            logits_per_text = logit_scale * (norm_text @ image_all.t())
            logits_per_image = logit_scale * (norm_images @ text_all.t())

            local_pron = src_input["pronunciation_batch"]
            global_pron = utils.all_gather_object_list(local_pron)
            labels = torch.zeros_like(logits_per_text)

            for i, pronunciation_i in enumerate(local_pron):
                for j, pronunciation_j in enumerate(global_pron):
                    if pronunciation_i == pronunciation_j:
                        labels[i, j] = 1.0

            loss_text = F.binary_cross_entropy_with_logits(logits_per_text, labels, pos_weight=torch.tensor(1, device=logits_per_text.device))
            loss_image = F.binary_cross_entropy_with_logits(logits_per_image, labels, pos_weight=torch.tensor(1, device=logits_per_image.device))
            total_loss = (loss_text + loss_image) / 2

        loss_scaler(total_loss, optimizer)

        if not math.isfinite(total_loss.item()):
            print(f"Loss is {total_loss.item()}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if args.run:
        args.run.log({"epoch": epoch + 1, "epoch/train_loss": total_loss.item()})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def evaluate(args, dataloader, model, criterion, epoch):
    """Computes retrieval metrics for ProSign_pretrain."""

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    with torch.no_grad():
        image_batches = []
        text_batches = []
        pronunciation_list = []

        for src_input, tgt_input in metric_logger.log_every(dataloader, 10, "Eval"):
            norm_text, norm_images = model(src_input, tgt_input)
            text_all = utils.all_gather_batch(norm_text)
            image_all = utils.all_gather_batch(norm_images)
            logit_scale = getattr(model, "module", model).logit_scale.exp().clamp(max=30)
            logits_per_text = logit_scale * (norm_text @ image_all.t())
            logits_per_image = logit_scale * (norm_images @ text_all.t())

            local_pron = src_input["pronunciation_batch"]
            global_pron = utils.all_gather_object_list(local_pron)
            labels = torch.zeros_like(logits_per_text)

            for i, pronunciation_i in enumerate(local_pron):
                for j, pronunciation_j in enumerate(global_pron):
                    if pronunciation_i == pronunciation_j:
                        labels[i, j] = 1.0

            loss_text = F.binary_cross_entropy_with_logits(logits_per_text, labels, pos_weight=torch.tensor(1, device=logits_per_text.device))
            loss_image = F.binary_cross_entropy_with_logits(logits_per_image, labels, pos_weight=torch.tensor(1, device=logits_per_image.device))
            total_loss = (loss_text + loss_image) / 2

            image_batches.append(norm_images)
            text_batches.append(norm_text)
            pronunciation_list.extend(local_pron)
            metric_logger.update(loss=total_loss.item())

    total_image_feats = utils.all_gather_batch(torch.cat(image_batches, dim=0))[: len(dataloader.dataset)]
    total_text_feats = utils.all_gather_batch(torch.cat(text_batches, dim=0))[: len(dataloader.dataset)]
    total_pronunciations = utils.all_gather_object_list(pronunciation_list)[: len(dataloader.dataset)]

    sim_matrix = torch.matmul(total_image_feats, total_text_feats.t())
    results = utils.compute_recall_multi_positive(
        sim_matrix,
        query_pronunciations=total_pronunciations,
        target_pronunciations=total_pronunciations,
    )

    metric_logger.meters["top1"].update(results["R1"])
    metric_logger.meters["top5"].update(results["R5"])
    metric_logger.meters["top10"].update(results["R10"])

    if args.run:
        args.run.log({"epoch": epoch + 1, "epoch/val_loss": total_loss.item()})

    metric_logger.synchronize_between_processes()
    print(
        "* Loss {losses.global_avg:.3f} | Top1 {top1.global_avg:.3f} | Top5 {top5.global_avg:.3f} | Top10 {top10.global_avg:.3f}".format(
            losses=metric_logger.loss,
            top1=metric_logger.top1,
            top5=metric_logger.top5,
            top10=metric_logger.top10,
        )
    )
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def setup_run(args, config):
    """Initializes the experiment tracker when logging is enabled."""

    if args.log_all:
        os.environ["WANDB_MODE"] = config["training"]["wandb"] if not args.eval else "disabled"
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split("/")[-1],
            config=config,
        )
    elif utils.is_main_process():
        os.environ["WANDB_MODE"] = config["training"]["wandb"] if not args.eval else "disabled"
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            config=config,
        )
        run.name = args.output_dir.split("/")[-1]
    else:
        os.environ["WANDB_MODE"] = "disabled"
        return False

    run.define_metric("epoch")
    run.define_metric("training/*", step_metric="epoch")
    run.define_metric("validation/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    return run


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("ProSign_pretrain", parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    args.run = setup_run(args, config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)

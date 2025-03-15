import logging
import shutil
import safetensors
import itertools
from pathlib import Path
from accelerate.logging import get_logger
import torch
import torch as th
import torch.nn.functional as F
import os
from demucs.states import load_model
import accelerate
import torchaudio as ta
import random
import json
import pickle
import argparse
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from demucs.htdemucs import HTDemucs
from mydata import MixDataset, SR, SLEN, FileIter, get_audio_files
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import shlex

logger = get_logger(__name__)

def init_model(pretrain_path):
    with open("kwargs.pkl", 'rb') as f:
        kwargs = pickle.load(f)
    model = HTDemucs(**kwargs)
    if pretrain_path:
        logger.info(f"load pretrained from {pretrain_path}")
        if pretrain_path.endswith(".safetensors"):
            logger.info(f"load safetensors")
            ckpt = safetensors.safe_open(pretrain_path, "pt")
            pretrained_state = {
                key: ckpt.get_tensor(key) for key in ckpt.keys()
            }
        else:
            with open(pretrain_path, "rb") as f:
                pretrained = torch.load(f, weights_only=False)
                pretrained_state = pretrained['state']
        valid_state = {}
        for name, param in model.named_parameters():
            if name in pretrained_state and param.shape == pretrained_state[name].shape:
                valid_state[name] = pretrained_state[name]
        logger.info(f"Loaded {len(valid_state)} params from {pretrain_path}")
        model.load_state_dict(valid_state, strict=False)
    return model


def get_optimizer(model, lr):
    params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
    groups = [
        {'params': params, "lr": lr}
    ]
    
    return torch.optim.Adam(
        groups,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0,
    )


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


#DS = "/mnt/e/dataset"

def collate_fn(examples):
    mixes = torch.stack([ex[0] for ex in examples], dim=0)
    main_segs = torch.stack([ex[1] for ex in examples], dim=0)
    noise_segs = torch.stack([ex[2] for ex in examples], dim=0)
    return mixes, main_segs, noise_segs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--gradient_checkpoint", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int)
    parser.add_argument("--validate_steps", type=int)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--lr", type=float)
    #args = parser.parse_args(shlex.split("--batch_size 4 --gradient_accumulation_steps 4 --output_dir output"))
    args = parser.parse_args()

    DS = args.ds
    print("DS", DS)

    logging_dir = Path(args.output_dir, "log")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        tracker_name = "train_orchestra"
        accelerator.init_trackers(tracker_name, config=vars(args))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    model = init_model(args.init_from)
    if args.gradient_checkpoint:
        model.gradient_checkpointing = True
    optimizer = get_optimizer(model, args.lr)
    params_to_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=-1)

    train_dataset = MixDataset(
        ds_root=DS,
        main=["orchestra", "cello"], noise="violin", cat="train", ex_cat="val", shift=True, shuffle=True,
        main_nt=4,
        noise_nt=3,    
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, collate_fn=collate_fn)
    val_dataset = MixDataset(
        ds_root=DS,
        main="orchestra", noise="violin", cat="val", shift=False, shuffle=False,
        main_nt=1,
        noise_nt=1,
        ex_cat=None,
        crop_noise=0, main_dropout=0, noise_dropout=0, random_weight=False,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, collate_fn=collate_fn)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    monitor_params = [p for p in model.named_parameters()]
    monitor_names = {monitor_params[i][0] for i in [1,4,9]}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print("NO GRAD", name)

    def print_params(cur_model):
        print()
        for name, params in cur_model.named_parameters():
            if name in monitor_names:
                print(name, params.sum().detach().cpu().item(), params.reshape(-1)[:3].detach().cpu().numpy())

    global_step = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        print("BEFORE RESTORE")
        print_params(model)
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            sd = lr_scheduler.state_dict()
            sd['base_lrs'] = [args.lr for _ in sd['base_lrs']]
            lr_scheduler.load_state_dict(sd)
        print("RESTORED GS", global_step)
        print_params(model)
    else:
        initial_global_step = 0

    # TODO: resume_from_checkpoint
    progress_bar = tqdm(
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def save_checkpoint():
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def validate():
        total_val_loss = 0
        total_val_batch = 0
        logger.info("validate")
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                total_val_batch += 1
                mix, label, noise = batch
                pred = model(mix.to(accelerator.device))
                diff = pred[:, 0]-label.to(accelerator.device)
                loss = torch.mean(diff**2)
                total_val_loss += loss.detach().cpu().item()
        val_loss = total_val_loss / total_val_batch
        logs = {"val/loss": val_loss}
        logger.info(f"val loss {val_loss}, step {global_step}")
        accelerator.log(logs, step=global_step)
    
    def evaluate():
        dest = os.path.join(args.output_dir, "eval_" + str(global_step))
        os.makedirs(dest, exist_ok=True)
        logger.info("evaludate")
        for fn in tqdm(get_audio_files(os.path.join(DS, "eval"))):
            fiter = FileIter.get_file_iter(fn, SLEN, False)
            if fiter is None:
                logger.info(f"SKIP {fn}")
                continue
            pred_tensors = []
            batch_tensors = []
            def pred():
                with torch.no_grad():
                    batch = torch.stack(batch_tensors, axis=0).to(accelerator.device)
                    pred = model(batch)[:, 0]
                    pred = pred.permute(1, 0, 2).reshape(2, -1).detach().cpu()
                    pred_tensors.append(pred)
                    
            while fiter.has_next():
                seg = fiter.next()
                batch_tensors.append(seg)
                if len(batch_tensors) == args.batch_size:
                    pred()
                    batch_tensors = []
            if len(batch_tensors) > 0:
                pred()
            if len(pred_tensors) > 0:
                pred = torch.concat(pred_tensors, axis=1)
                basename = os.path.basename(fn)
                ta.save(os.path.join(dest, basename), pred, SR)
    if args.init_from:
        print("INITIAL EVAL")
        validate()
        evaluate()

    for epoch in range(100):
        print("EPOCH", epoch)
        models_to_accumulate = [model]
        total_loss = 0
        total_l2_loss = 0
        total_l1_loss = 0
        print_params(model)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models_to_accumulate):
                mix, label, noise = batch
                pred = model(mix.to(accelerator.device))
                l2_loss = torch.nn.functional.mse_loss(pred[:, 0], label.to(accelerator.device))
                l1_loss = torch.nn.functional.l1_loss(pred[:, 0], label.to(accelerator.device))

                loss = l2_loss + l1_loss / 100
                # if args.loss == "l1":
                #     loss = l1_loss
                # else:
                #     loss = l2_loss

                total_loss += loss.detach().cpu().item()
                total_l2_loss += l2_loss.detach().cpu().item()
                total_l1_loss += l1_loss.detach().cpu().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                total_loss /= args.gradient_accumulation_steps
                total_l2_loss /= args.gradient_accumulation_steps
                total_l1_loss /= args.gradient_accumulation_steps
                logs = {"loss": total_loss, "l1_loss": total_l1_loss, "l2_loss": total_l2_loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(epoch=epoch, **logs)
                accelerator.log(logs, step=global_step)
                total_loss = 0
                total_l2_loss = 0
                total_l1_loss = 0
                
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % 100 == 0:
                        print_params(accelerator.unwrap_model(model))
                    if global_step % args.checkpointing_steps == 0:
                        save_checkpoint()
                
                if global_step % args.validate_steps == 0:
                    if accelerator.is_main_process:
                        validate()
                        evaluate()
                    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
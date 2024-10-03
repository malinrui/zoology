import argparse
import random
from datetime import datetime
from typing import List, Union
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data
from zoology.config import TrainConfig
from zoology.mixers.mamba import Mamba, AnotherMamba, FakeMamba
from zoology.model import LanguageModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism

from torch.nn import functional as F


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
        load_from_pretrained_path: str = None,
        mix_with_mamba: bool = False,
        mamba_layers: List[int] = None,
        init_from_attention_weights: bool = False,
        freeze_attn: bool = False,
        fake_mamba: bool = False,

        teacher_model: LanguageModel = None,
        teacher_model_path: str = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys

        self.teacher_model = teacher_model

        if load_from_pretrained_path is not None:
            self.model.load_state_dict(torch.load(load_from_pretrained_path))
            print(f"******************* Model loaded from {load_from_pretrained_path} *******************")

        if mix_with_mamba:
            print("******************* Mixing with Mamba *******************")
            for layer_idx in mamba_layers:
                layer_encoder = AnotherMamba(
                    128,
                    128,
                    128,
                )

                if init_from_attention_weights:
                    attn_layer = self.model.backbone.layers[layer_idx].sequence_mixer
                    print("attn_layer.Wqkv.state_dict().shape:", attn_layer.Wqkv.state_dict()['weight'].shape)
                    layer_encoder.in_proj_x.load_state_dict(
                        {
                            'weight': attn_layer.Wqkv.state_dict()['weight'][2*128:3*128, :],
                            'bias': attn_layer.Wqkv.state_dict()['bias'][2*128:3*128],
                        }
                    )
                    layer_encoder.B_proj.load_state_dict(
                        {
                            'weight': attn_layer.Wqkv.state_dict()['weight'][128:2*128, :],
                        }
                    )
                    layer_encoder.C_proj.load_state_dict(
                        {
                            'weight': attn_layer.Wqkv.state_dict()['weight'][0:128, :],
                        }
                    )
                    layer_encoder.out_proj.load_state_dict(
                        attn_layer.out_proj.state_dict()
                    )

                    print('@@@@@@@layer ' + str(layer_idx) + ' init_from_attention_weights!@@@@@@@')


                self.model.backbone.layers[layer_idx].sequence_mixer = layer_encoder
            # self.model.backbone.layers[0].sequence_mixer = Mamba(128)
            print("******************* Model mixed with Mamba *******************")
            print("######################################################")
            print("######################################################")

            print("model NOW:", self.model)

            print("######################################################")
            print("######################################################")

            if init_from_attention_weights:
                print("@@@@@@@@@@@@@@@@@@@@@@ Initializing Mamba with attention weights @@@@@@@@@@@@@@@@@@@@@@")

            if freeze_attn:
                print("$$$$$$$$$$$$$$$$$$$$$$ Freezing attention weights $$$$$$$$$$$$$$$$$$$$$$")
                for name, param in self.model.named_parameters():
                    # print(name)
                    if f"sequence_mixer" not in name:
                        param.requires_grad = False

            if fake_mamba:
                for layer_idx in mamba_layers:
                    layer_encoder = FakeMamba()
                    self.model.backbone.layers[layer_idx].sequence_mixer = layer_encoder
                print("\n%%%%%%%%%%%%%%%%%%%%% OOPS! It's Fake Mamba(nn.Identity()) %%%%%%%%%%%%%%%%%%%%%\n")
                print("Faked model NOW:", self.model)

        if teacher_model is not None:
            self.teacher_model.load_state_dict(torch.load(teacher_model_path))
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.to("cuda")

            print(f"\n******************* Teacher Model loaded from {teacher_model_path} *******************\n")


    def train_epoch_with_teacher(self, epoch_idx: int):
        self.model.train()
        self.teacher_model.eval()

        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, true_target, slices in iterator:
            inputs, true_target = inputs.to(self.device), true_target.to(self.device)
            self.optimizer.zero_grad()

            # forward
            student_logits = self.model(inputs)

            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)

            teacher_ce_loss = self.loss_fn(
                rearrange(teacher_logits, "... c -> (...) c"), true_target.flatten()
            )

            # 选择teacher的logits中最大者作为target，直接使target.shape [64, 256, 512] - >[64, 256]
            pseudo_targets = torch.argmax(teacher_logits, dim=-1)

            # print('targets shape:', targets.shape)

            main_loss = self.loss_fn(
                rearrange(student_logits, "... c -> (...) c"), pseudo_targets.flatten()
            )

            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            )


            true_target_loss = self.loss_fn(
                rearrange(student_logits, "... c -> (...) c"), true_target.flatten()
            )

            # collect auxiliary losses
            auxiliary_loss = []

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            self.model.apply(get_auxiliary_loss)
            auxiliary_loss = sum(auxiliary_loss)

            loss = 0.1 * kl_loss + 1 * (main_loss + auxiliary_loss)

            loss.backward()
            self.optimizer.step()

            # logging and printing
            iterator.set_postfix({"loss": loss.item(), 'true_target_loss': true_target_loss.item(), 'teacher_ce_loss': teacher_ce_loss.item()})
            self.logger.log(
                {
                    "train/loss": loss,
                    "train/main_loss": main_loss,
                    "train/auxiliary_loss": auxiliary_loss,
                    "train/kl_loss": kl_loss,
                    "train/teacher_ce_loss": teacher_ce_loss,
                    "train/true_target_loss": true_target_loss,
                    "epoch": epoch_idx,
                }
            )

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            logits = self.model(inputs)

            # collect auxiliary losses
            auxiliary_loss = []

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            self.model.apply(get_auxiliary_loss)
            auxiliary_loss = sum(auxiliary_loss)

            # need to flatten batch and sequence dimensions
            main_loss = self.loss_fn(
                rearrange(logits, "... c -> (...) c"), targets.flatten()
            )
            loss = main_loss + auxiliary_loss
            loss.backward()
            self.optimizer.step()

            # logging and printing
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log(
                {
                    "train/loss": loss,
                    "train/main_loss": main_loss,
                    "train/auxiliary_loss": auxiliary_loss,
                    "epoch": epoch_idx,
                }
            )

    def test(self, epoch_idx: int, teacher=False):
        self.model.eval()

        if teacher:
            self.teacher_model.eval()

        test_loss = 0
        # all_preds = []
        # all_targets = []
        results = [] 

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if teacher:
                    logits = self.teacher_model(inputs)
                else:
                    logits = self.model(inputs)

                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), targets.flatten()
                )
                test_loss += loss / len(self.test_dataloader)

                # SE: important to
                preds = torch.argmax(logits, dim=-1).cpu()
                results.extend(compute_metrics(preds, targets.cpu(), slices))
               
                iterator.update(1)

            # test_accuracy = compute_accuracy(
            #     torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
            # )
            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            if not teacher:
                self.logger.log({"epoch": epoch_idx, **metrics})
        return metrics

    def fit(self):
        self.model.to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), #一点加速 27iter/s -> 32iter/s
            # self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )

        print("Testing the raw model in the first epoch:")
        self.test(-1)

        if self.teacher_model is not None:
            print("\nTesting the teacher model in the first epoch:")
            self.teacher_model.to("cuda")
            self.test(-1, teacher=True)

        print("\nNow training...")

        for epoch_idx in range(self.max_epochs):
            if self.teacher_model is not None:
                self.train_epoch_with_teacher(epoch_idx)
            else:
                self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    # TODO (SE): need to actaully verify reproducibility here
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config=config.model)

    print("######################################################")
    print("######################################################")

    print("model:", model)

    print("######################################################")
    print("######################################################")
    
    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        load_from_pretrained_path=config.load_from_pretrained_path,
        mix_with_mamba=config.mix_with_mamba,
        mamba_layers=config.mamba_layers,
        init_from_attention_weights=config.init_from_attention_weights,
        freeze_attn=config.freeze_attn,
        fake_mamba=config.fake_mamba,

        teacher_model=LanguageModel(config=config.model) if config.teacher_model_path is not None else None,
        teacher_model_path=config.teacher_model_path,
    )
    task.fit()

    if config.save_model:
        save_path = f"trained_models/{config.run_id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train(config)

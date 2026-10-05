import os
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from data_displacement_cached import DisplacementDatasetCached
from model_reg_U import PctRegressor


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class RegressionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)

        self._init_dataloaders()
        self._save_norm_params()
        self._save_run_config()
        self._init_model()

        self.ema = EMA(self.model, decay=self.args.ema_decay) if self.args.use_ema else None
        self._init_optim()

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self.best_rmse_mm = float('inf')
        self.best_mae_mm = float('inf')

        self.log_df = pd.DataFrame(columns=[
            'Epoch',
            'Train SmoothL1 (norm)',
            'Train MSE (norm)',
            'Val MAE (mm)',
            'Val RMSE (mm)',
            'Val WMAPE',
            'Learning Rate',
            'Avg Grad Norm',
            'Max Grad Norm'
        ])
        self.log_path = f"checkpoints/{args.exp_name}/training_log.xlsx"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _save_norm_params(self):
        norm_path = f"checkpoints/{self.args.exp_name}/norm_params.json"
        norm = {
            "cloud_mean": self.train_set.cloud_mean.tolist(),
            "cloud_std": self.train_set.cloud_std.tolist(),
            "disp_mean": self.train_set.disp_mean.tolist(),
            "disp_std": self.train_set.disp_std.tolist(),
        }
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(norm, f, indent=2, ensure_ascii=False)
        print(f"归一化参数已保存: {norm_path}")

    def _save_run_config(self):
        cfg_path = f"checkpoints/{self.args.exp_name}/config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)
        print(f"超参数已保存: {cfg_path}")

    def _init_dataloaders(self):
        self.train_set = DisplacementDatasetCached(
            root_dir=self.args.data_root,
            split=self.args.train_split,
            is_training_split=True,
            cache_root=self.args.cache_root,
            num_points=self.args.num_points,
            augment=self.args.augment,
            seed=self.args.seed,
        )

        self.val_set = DisplacementDatasetCached(
            root_dir=self.args.data_root,
            split=self.args.val_split,
            is_training_split=False,
            cache_root=self.args.cache_root,
            num_points=self.args.num_points,
            disp_mean=self.train_set.disp_mean,
            disp_std=self.train_set.disp_std,
            cloud_mean=self.train_set.cloud_mean,
            cloud_std=self.train_set.cloud_std,
            augment=False,
            seed=self.args.seed,
        )

        if self.val_set.num_nodes != self.train_set.num_nodes:
            raise RuntimeError(
                f"train/val num_nodes 不一致: train={self.train_set.num_nodes}, val={self.val_set.num_nodes}"
            )

        nw = int(self.args.num_workers)
        pin_mem = (self.device.type == "cuda")

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin_mem,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin_mem,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

        print(
            f"训练集: {len(self.train_set)} 样本 (split={self.args.train_split}, augment={'on' if self.args.augment else 'off'}) | "
            f"验证集: {len(self.val_set)} 样本 (split={self.args.val_split})"
        )

    def _init_model(self):
        self.model = PctRegressor(self.args, num_nodes=self.train_set.num_nodes).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型初始化完成 | 总参数: {total_params:,} | 可训练参数: {trainable_params:,}")

    def _init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            eps=1e-6,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            min_lr=self.args.min_lr,
            verbose=True
        )

    def _compute_metrics(self, pred_norm, target_norm):
        pred_flat = pred_norm.reshape(pred_norm.size(0), -1)
        tgt_flat = target_norm.reshape(target_norm.size(0), -1)

        mse_norm = F.mse_loss(pred_flat, tgt_flat)
        mae_norm = F.l1_loss(pred_flat, tgt_flat)

        diff_norm = pred_norm - target_norm

        std = torch.as_tensor(
            self.train_set.disp_std,
            dtype=diff_norm.dtype,
            device=diff_norm.device
        ).view(1, 1, 3)

        mean = torch.as_tensor(
            self.train_set.disp_mean,
            dtype=diff_norm.dtype,
            device=diff_norm.device
        ).view(1, 1, 3)

        # 误差反归一化：均值会相互抵消，所以只需要乘 std
        diff_mm = diff_norm * std

        # 真实位移反归一化：必须恢复到原始物理量空间
        target_mm = target_norm * std + mean

        mae_mm = diff_mm.abs().mean().item()
        mse_mm2 = diff_mm.pow(2).mean().item()
        rmse_mm = math.sqrt(mse_mm2)

        abs_err_sum_mm = diff_mm.abs().sum().item()
        abs_true_sum_mm = target_mm.abs().sum().item()
        wmape = abs_err_sum_mm / max(abs_true_sum_mm, 1e-12)

        return {
            'mse_norm': float(mse_norm.item()),
            'mae_norm': float(mae_norm.item()),
            'mae_mm': float(mae_mm),
            'mse_mm2': float(mse_mm2),
            'rmse_mm': float(rmse_mm),
            'abs_err_sum_mm': float(abs_err_sum_mm),
            'abs_true_sum_mm': float(abs_true_sum_mm),
            'wmape': float(wmape),
            'bsz': int(pred_norm.size(0)),
        }

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss_norm = 0.0
        epoch_mse_norm = 0.0
        total_bsz = 0
        grad_norms = []
        num_batches = len(self.train_loader)

        for batch_idx, (clouds, disps, nodes) in enumerate(self.train_loader):
            if torch.isnan(clouds).any() or torch.isnan(disps).any():
                print("[WARN] 跳过包含 NaN 的训练 batch")
                continue

            clouds = clouds.float().to(self.device, non_blocking=True)
            disps = disps.float().to(self.device, non_blocking=True)
            nodes = nodes.float().to(self.device, non_blocking=True)

            with autocast(enabled=self.args.amp):
                preds = self.model(clouds, nodes)
                loss = F.smooth_l1_loss(
                    preds, disps,
                    reduction='mean',
                    beta=self.args.huber_beta
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args.grad_clip,
            )
            grad_norms.append(float(grad_norm.item()))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            metrics = self._compute_metrics(preds, disps)
            bsz = metrics['bsz']

            epoch_loss_norm += float(loss.item()) * bsz
            epoch_mse_norm += metrics['mse_norm'] * bsz
            total_bsz += bsz

            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{num_batches} | "
                    f"LR: {current_lr:.2e} | SmoothL1: {float(loss.item()):.4f} | AvgGrad: {avg_grad:.2f}"
                )

        avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        max_grad = float(np.max(grad_norms)) if grad_norms else 0.0

        train_loss_norm = epoch_loss_norm / max(1, total_bsz)
        train_mse_norm = epoch_mse_norm / max(1, total_bsz)

        return train_loss_norm, train_mse_norm, avg_grad, max_grad

    def _validate(self):
        self.model.eval()
        use_ema = (self.ema is not None)
        if use_ema:
            self.ema.apply_to(self.model)

        total_mse_norm = 0.0
        total_mae_mm = 0.0
        total_mse_mm2 = 0.0
        total_abs_err_sum_mm = 0.0
        total_abs_true_sum_mm = 0.0
        total_bsz = 0

        with torch.no_grad():
            for clouds, disps, nodes in self.val_loader:
                if torch.isnan(clouds).any() or torch.isnan(disps).any():
                    print("[WARN] 跳过包含 NaN 的验证 batch")
                    continue

                clouds = clouds.float().to(self.device, non_blocking=True)
                disps = disps.float().to(self.device, non_blocking=True)
                nodes = nodes.float().to(self.device, non_blocking=True)

                preds = self.model(clouds, nodes)
                if torch.isnan(preds).any():
                    print("[WARN] 验证预测包含 NaN，跳过该 batch")
                    continue

                metrics = self._compute_metrics(preds, disps)
                bsz = metrics['bsz']

                total_mse_norm += metrics['mse_norm'] * bsz
                total_mae_mm += metrics['mae_mm'] * bsz
                total_mse_mm2 += metrics['mse_mm2'] * bsz
                total_abs_err_sum_mm += metrics['abs_err_sum_mm']
                total_abs_true_sum_mm += metrics['abs_true_sum_mm']
                total_bsz += bsz

        if total_bsz == 0:
            if use_ema:
                self.ema.restore(self.model)
            return {
                'mse_norm': float('inf'),
                'mae_mm': float('inf'),
                'rmse_mm': float('inf'),
                'wmape': float('inf')
            }

        mse_norm = total_mse_norm / total_bsz
        mae_mm = total_mae_mm / total_bsz
        mse_mm2 = total_mse_mm2 / total_bsz
        rmse_mm = math.sqrt(mse_mm2)
        wmape = total_abs_err_sum_mm / max(total_abs_true_sum_mm, 1e-12)

        if use_ema:
            self.ema.restore(self.model)

        return {
            'mse_norm': mse_norm,
            'mae_mm': mae_mm,
            'rmse_mm': rmse_mm,
            'wmape': wmape
        }

    def train(self):
        for epoch in range(self.args.epochs):
            start_time = time.time()

            train_loss_norm, train_mse_norm, avg_grad, max_grad = self._train_epoch(epoch)
            val_metrics = self._validate()

            self.scheduler.step(val_metrics["rmse_mm"])

            if val_metrics['rmse_mm'] < self.best_rmse_mm:
                self.best_rmse_mm = val_metrics['rmse_mm']
                self.best_mae_mm = val_metrics['mae_mm']
                torch.save(self.model.state_dict(), f"checkpoints/{self.args.exp_name}/best_model.pth")
                print(f"保存新最佳模型 | Val RMSE: {self.best_rmse_mm:.4f} mm | Val MAE: {self.best_mae_mm:.4f} mm")

            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch + 1}/{self.args.epochs} | Time: {epoch_time:.1f}s")
            print(f"Train SmoothL1 (norm): {train_loss_norm:.6f}")
            print(f"Train MSE (norm): {train_mse_norm:.6f}")
            print(
                f"Val MAE: {val_metrics['mae_mm']:.4f} mm | "
                f"Val RMSE: {val_metrics['rmse_mm']:.4f} mm | "
                f"Val WMAPE: {val_metrics['wmape']:.6f} ({val_metrics['wmape'] * 100:.2f}%) | "
                f"Best RMSE: {self.best_rmse_mm:.4f} mm\n"
            )

            new_log = pd.DataFrame({
                'Epoch': [epoch + 1],
                'Train SmoothL1 (norm)': [train_loss_norm],
                'Train MSE (norm)': [train_mse_norm],
                'Val MAE (mm)': [val_metrics['mae_mm']],
                'Val RMSE (mm)': [val_metrics['rmse_mm']],
                'Val WMAPE': [val_metrics['wmape']],
                'Learning Rate': [lr],
                'Avg Grad Norm': [avg_grad],
                'Max Grad Norm': [max_grad],
            })
            self.log_df = pd.concat([self.log_df, new_log], ignore_index=True)

            if (epoch + 1) % 5 == 0 or epoch == self.args.epochs - 1:
                self.log_df.to_excel(self.log_path, index=False)
                print(f"训练日志已保存至 {self.log_path}")

        self.log_df.to_excel(self.log_path, index=False)
        return self.log_df


def get_args():
    parser = argparse.ArgumentParser(description="点云位移场回归训练")

    parser.add_argument('--data_root', type=str, default='data400_new')
    parser.add_argument('--train_split', type=str, default='train', help='训练集子目录名')
    parser.add_argument('--val_split', type=str, default='test', help='验证集子目录名；若你有 val 目录，改成 val 更严谨')
    parser.add_argument('--num_points', type=int, default=8944)
    parser.add_argument('--cache_root', type=str, default=None, help='缓存基目录；实际读取 <cache_root>/<split>/...')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_point_tokens', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--huber_beta', type=float, default=0.5)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--exp_name', type=str, default='my_exp')
    parser.add_argument('--augment', action='store_true', help='是否在训练时纳入 *_augXX 增强缓存样本')

    return parser.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('high')

    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n===== 训练配置 =====")
    print(f"实验名称: {args.exp_name}")
    print(f"训练设备: {'GPU' if (not args.no_cuda and torch.cuda.is_available()) else 'CPU'}")
    print(f"混合精度: {'启用' if args.amp else '禁用'}")
    print(f"训练损失: SmoothL1(beta={args.huber_beta})")
    print(f"训练 split: {args.train_split}")
    print(f"验证 split: {args.val_split}")
    print(f"数据增强: {'启用' if args.augment else '禁用'}")
    print(f"点 token 数: {args.num_point_tokens}")
    print("====================\n")

    trainer = RegressionTrainer(args)
    trainer.train()

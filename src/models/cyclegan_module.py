from typing import Any, Dict, Tuple, Type

import itertools
from time import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning import LightningModule
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics import MeanMetric


from .gan.loss import GANLoss
from .utils import set_requires_grad, init_weights
from ..utils.visualization import concat_images
from ..utils.image_pool import ImagePool
from ..data.ab_data_processor import ABDataProcessor


class CycleGANLitModule(LightningModule):

    def __init__(
        self,
        G_net: Type[torch.nn.Module],
        D_net: Type[torch.nn.Module],
        optimizer: Type[torch.optim.Optimizer],
        scheduler: Type[Any],
        compile: bool,
        gan_mode: str,
        lambda_identity: float = 0.5,
        lambda_A: float = 10.0,
        lambda_B: float = 10.0,
        pool_size: int = 50,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False

        self.G_A_net = G_net()  # A -> B
        self.G_B_net = G_net()  # B -> A

        self.D_A_net = None
        self.D_B_net = None

        self.MAX_SAMPLES = 16
        self.A_to_B_samples = []
        self.B_to_A_samples = []

        self.val_A_fid = FrechetInceptionDistance(feature=2048)
        self.val_B_fid = FrechetInceptionDistance(feature=2048)

        if self.training:
            self.D_A_net = D_net()
            self.D_B_net = D_net()

            init_weights(self.D_A_net)
            init_weights(self.D_B_net)

            self.lambda_identity = lambda_identity
            self.lambda_A = lambda_A
            self.lambda_B = lambda_B

            self.fake_A_pool = ImagePool(pool_size)
            self.fake_B_pool = ImagePool(pool_size)

            self.gan_loss = GANLoss(gan_mode=gan_mode)

    # def on_train_start(self) -> None:
    #     pass

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        G_optimizer, D_A_optimizer, D_B_optimizer = self.optimizers()

        last_time = time()

        real_A, real_B = batch["A_image"], batch["B_image"]

        fake_B = self.G_A_net(real_A)
        fake_A = self.G_B_net(real_B)
        rec_A = self.G_B_net(fake_B)
        rec_B = self.G_A_net(fake_A)

        # generator loss
        # set_requires_grad([self.D_A_net, self.D_B_net], False)
        G_optimizer.optimizer.zero_grad()

        identity_A = self.G_A_net(real_B)
        identity_B = self.G_B_net(real_A)

        identity_A_loss = F.l1_loss(identity_A, real_B)
        identity_B_loss = F.l1_loss(identity_B, real_A)

        G_A_loss = self.gan_loss(self.D_A_net(fake_B), True)
        G_B_loss = self.gan_loss(self.D_B_net(fake_A), True)

        cycle_A_loss = F.l1_loss(rec_A, real_A)
        cycle_B_loss = F.l1_loss(rec_B, real_B)

        identity_loss = (identity_A_loss + identity_B_loss) / 2
        gan_loss = (G_A_loss + G_B_loss) / 2
        cycle_loss = (cycle_A_loss + cycle_B_loss) / 2

        G_loss = (
            G_A_loss
            + G_B_loss
            + identity_A_loss * self.lambda_identity * self.lambda_A
            + identity_B_loss * self.lambda_identity * self.lambda_B
            + cycle_A_loss * self.lambda_A
            + cycle_B_loss * self.lambda_B
        )

        self.manual_backward(G_loss)

        G_optimizer.step()

        # # discriminator loss
        # set_requires_grad([self.D_A_net, self.D_B_net], True)
        D_A_optimizer.optimizer.zero_grad()

        # real A
        real_A_pred = self.D_A_net(real_A)
        loss_D_A_real = self.gan_loss(real_A_pred, True)

        # fake A
        fake_A = self.fake_A_pool.query(fake_A)
        fake_A_pred = self.D_A_net(fake_A.detach())
        loss_D_A_fake = self.gan_loss(fake_A_pred, False)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2
        self.manual_backward(loss_D_A)
        D_A_optimizer.optimizer.step()

        D_B_optimizer.optimizer.zero_grad()

        # real A
        real_B_pred = self.D_B_net(real_B)
        loss_D_B_real = self.gan_loss(real_B_pred, True)

        # fake A
        fake_B = self.fake_B_pool.query(fake_B)
        fake_B_pred = self.D_B_net(fake_B.detach())
        loss_D_B_fake = self.gan_loss(fake_B_pred, False)

        loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2
        self.manual_backward(loss_D_B)
        D_B_optimizer.step()

        self.log_dict(
            {
                "train/G_loss": G_loss,
                "train/identity_loss": identity_loss,
                "train/gan_loss": gan_loss,
                "train/cycle_loss": cycle_loss,
                "train/D_A_loss": loss_D_A,
                "train/D_B_loss": loss_D_B,
            },
            prog_bar=True,
        )

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("training_speed", 1 / (time() - last_time))

    def on_train_epoch_end(self) -> None:
        G_scheduler, D_A_scheduler, D_B_scheduler = self.lr_schedulers()

        G_scheduler.step()
        D_A_scheduler.step()
        D_B_scheduler.step()

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        ab_processor = ABDataProcessor(256)

        if "A_image" in batch:
            real_A = batch["A_image"]
            fake_B = self.G_A_net(batch["A_image"])

            real_A = torch.stack(ab_processor.postprocess(real_A, to_pil=False))
            fake_B = torch.stack(ab_processor.postprocess(fake_B, to_pil=False))

            self.val_A_fid.update(real_A, real=True)
            self.val_B_fid.update(fake_B, real=False)

            if len(self.A_to_B_samples) < self.MAX_SAMPLES:
                self.A_to_B_samples.append(concat_images([real_A[0], fake_B[0]]))

        if "B_image" in batch:
            real_B = batch["B_image"]
            fake_A = self.G_B_net(batch["B_image"])

            real_B = torch.stack(ab_processor.postprocess(real_B, to_pil=False))
            fake_A = torch.stack(ab_processor.postprocess(fake_A, to_pil=False))

            self.val_B_fid.update(real_B, real=True)
            self.val_A_fid.update(fake_A, real=False)

            if len(self.B_to_A_samples) < self.MAX_SAMPLES:
                self.B_to_A_samples.append(concat_images([real_B[0], fake_A[0]]))

    def on_validation_epoch_end(self) -> None:
        val_A_fid = self.val_A_fid.compute()
        val_B_fid = self.val_B_fid.compute()

        self.log_dict(
            {
                "val/A_fid": val_A_fid,
                "val/B_fid": val_B_fid,
                "val/mean_fid": (val_A_fid + val_B_fid) / 2,
            },
            prog_bar=True,
        )

        if self.is_wandb_enabled():
            wandb.log(
                {
                    "day_to_night_samples": [
                        wandb.Image(image) for image in self.A_to_B_samples
                    ],
                    "night_to_day_samples": [
                        wandb.Image(image) for image in self.B_to_A_samples
                    ],
                }
            )

    def is_wandb_enabled(self) -> bool:
        if not isinstance(self.trainer.logger, list):
            return isinstance(self.trainer.logger, WandbLogger)

        for logger in self.trainer.logger:
            if isinstance(logger, WandbLogger):
                return True

        return False

    # def on_validation_epoch_end(self) -> None:
    #     pass

    # def test_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    # ) -> None:
    #     pass

    # def setup(self, stage: str) -> None:
    #     pass

    def configure_optimizers(self) -> Tuple[Any, Any]:
        G_optimizer = self.hparams.optimizer(
            params=itertools.chain(self.G_A_net.parameters(), self.G_B_net.parameters())
        )

        D_A_optimizer = self.hparams.optimizer(params=self.D_A_net.parameters())
        D_B_optimizer = self.hparams.optimizer(params=self.D_B_net.parameters())

        G_scheduler = self.hparams.scheduler(G_optimizer)
        D_A_scheduler = self.hparams.scheduler(D_A_optimizer)
        D_B_scheduler = self.hparams.scheduler(D_B_optimizer)

        return [G_optimizer, D_A_optimizer, D_B_optimizer], [
            G_scheduler,
            D_A_scheduler,
            D_B_scheduler,
        ]

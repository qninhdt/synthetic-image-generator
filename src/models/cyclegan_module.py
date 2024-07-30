from typing import Any, Dict, Tuple, Type

import itertools

import torch
import torch.nn.functional as F
from lightning import LightningModule
from hydra.utils import instantiate

from .gan.loss import GANLoss
from .utils import set_requires_grad, init_weights


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

        if self.training:
            self.D_A_net = D_net()
            self.D_B_net = D_net()

            init_weights(self.D_A_net)
            init_weights(self.D_B_net)

            self.lambda_identity = lambda_identity
            self.lambda_A = lambda_A
            self.lambda_B = lambda_B

            self.gan_loss = GANLoss(gan_mode=gan_mode)

    def on_train_start(self) -> None:
        pass

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        G_optimizer, D_optimizer = self.optimizers(use_pl_optimizer=False)

        real_A, real_B = batch["A_image"], batch["B_image"]

        fake_B = self.G_A_net(real_A)
        fake_A = self.G_B_net(real_B)
        rec_A = self.G_B_net(fake_B)
        rec_B = self.G_A_net(fake_A)

        # generator loss
        set_requires_grad([self.D_A_net, self.D_B_net], False)
        G_optimizer.zero_grad()

        identity_A = self.G_A_net(real_B)
        identity_B = self.G_B_net(real_A)

        identity_A_loss = F.l1_loss(identity_A, real_B) * self.lambda_identity
        identity_B_loss = F.l1_loss(identity_B, real_A) * self.lambda_identity

        G_A_loss = self.gan_loss(self.D_A_net(fake_B), True)
        G_B_loss = self.gan_loss(self.D_B_net(fake_A), True)

        cycle_A_loss = F.l1_loss(rec_A, real_A) * self.lambda_A
        cycle_B_loss = F.l1_loss(rec_B, real_B) * self.lambda_B

        G_loss = (
            G_A_loss
            + G_B_loss
            + cycle_A_loss
            + cycle_B_loss
            + identity_A_loss
            + identity_B_loss
        )

        self.manual_backward(G_loss)

        G_optimizer.step()

        # # discriminator loss
        set_requires_grad([self.D_A_net, self.D_B_net], True)
        D_optimizer.zero_grad()

        # real B
        real_B_pred = self.D_A_net(real_B)
        loss_D_A_real = self.gan_loss(real_B_pred, True)

        # fake B
        fake_B_pred = self.D_A_net(fake_B.detach())
        loss_D_A_fake = self.gan_loss(fake_B_pred, False)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        self.manual_backward(loss_D_A)

        # real A
        real_A_pred = self.D_B_net(real_A)
        loss_D_B_real = self.gan_loss(real_A_pred, True)

        # fake A
        fake_A_pred = self.D_B_net(fake_A.detach())
        loss_D_B_fake = self.gan_loss(fake_A_pred, False)

        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        self.manual_backward(loss_D_B)

        self.clip_gradients(
            D_optimizer, gradient_clip_val=0.1, gradient_clip_algorithm="norm"
        )
        D_optimizer.step()

        self.log_dict(
            {
                "train/G_A_loss": G_A_loss,
                "train/G_B_loss": G_B_loss,
                "train/cycle_A_loss": cycle_A_loss,
                "train/cycle_B_loss": cycle_B_loss,
                "train/identity_A_loss": identity_A_loss,
                "train/identity_B_loss": identity_B_loss,
                "train/D_A_loss": loss_D_A,
                "train/D_B_loss": loss_D_B,
            },
            prog_bar=True,
        )

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

    def on_train_epoch_end(self) -> None:
        G_scheduler, D_scheduler = self.lr_schedulers()

        G_scheduler.step()
        D_scheduler.step()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def configure_optimizers(self) -> Tuple[Any, Any]:
        G_optimizer = self.hparams.optimizer(
            params=itertools.chain(self.G_A_net.parameters(), self.G_B_net.parameters())
        )

        D_optimizer = self.hparams.optimizer(
            params=itertools.chain(self.D_A_net.parameters(), self.D_B_net.parameters())
        )
        G_scheduler = self.hparams.scheduler(G_optimizer)
        D_scheduler = self.hparams.scheduler(D_optimizer)

        return [G_optimizer, D_optimizer], [G_scheduler, D_scheduler]

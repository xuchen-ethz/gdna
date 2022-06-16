""" The code is based on https://github.com/apple/ml-gsn/ with adaption. """

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from lib.model.discriminator import StyleDiscriminator

def hinge_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        d_loss_real = torch.mean(F.relu(1.0 - real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = -torch.mean(fake_pred)
    return d_loss

def logistic_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.softplus(fake_pred))
        d_loss_real = torch.mean(F.softplus(-real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = torch.mean(F.softplus(-fake_pred))
    return d_loss


def r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class GANLoss(nn.Module):
    def __init__(
        self,
        opt,
        disc_loss='logistic',
    ):
        super().__init__()
        self.opt = opt

        input_dim = 3
        self.discriminator = StyleDiscriminator(input_dim, self.opt.img_res)

        if disc_loss == 'hinge':
            self.disc_loss = hinge_loss
        elif disc_loss == 'logistic':
            self.disc_loss = logistic_loss

    def forward(self, input, global_step, optimizer_idx):

        disc_in_real = input['norm_real']
        disc_in_fake = input['norm_fake']

        disc_in_real.requires_grad = True  # for R1 gradient penalty

        if optimizer_idx == 0:  # optimize generator
            loss = 0
            log = {}
            if self.opt.lambda_gan > 0:
                logits_fake = self.discriminator(disc_in_fake)
                g_loss = self.disc_loss(logits_fake, None, mode='g')
                log["loss_train/g_loss"] = g_loss.detach()
                loss += g_loss

            return loss, log

        if optimizer_idx == 1 :  # optimize discriminator
            logits_real = self.discriminator(disc_in_real)
            logits_fake = self.discriminator(disc_in_fake.detach().clone())

            disc_loss = self.disc_loss(fake_pred=logits_fake, real_pred=logits_real, mode='d')

            # lazy regularization so we don't need to compute grad penalty every iteration
            if (global_step % self.opt.d_reg_every == 0) and self.opt.lambda_grad > 0:
                grad_penalty = r1_loss(logits_real, disc_in_real)

                # the 0 * logits_real is to trigger DDP allgather
                # https://github.com/rosinality/stylegan2-pytorch/issues/76
                grad_penalty = self.opt.lambda_grad / 2 * grad_penalty * self.opt.d_reg_every + (0 * logits_real.sum())
            else:
                grad_penalty = torch.tensor(0.0).type_as(disc_loss)

            d_loss = disc_loss + grad_penalty #+ disc_recon_loss 

            log = {
                "loss_train/disc_loss": disc_loss.detach(),
                "loss_train/r1_loss": grad_penalty.detach(),
                "loss_train/logits_real": logits_real.mean().detach(),
                "loss_train/logits_fake": logits_fake.mean().detach(),
            }

            return d_loss*self.opt.lambda_gan, log


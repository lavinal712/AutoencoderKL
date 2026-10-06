import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )
    return d_loss


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def non_saturating_g_loss(logits_fake):
    return torch.mean(F.softplus(-logits_fake))

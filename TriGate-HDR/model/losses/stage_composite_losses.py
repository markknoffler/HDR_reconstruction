from .reconstruction_losses import gated_l2_loss
from .distribution_losses import wasserstein_hist_loss
from .structure_losses import structural_fidelity_loss, seam_gradient_continuity_loss
from .material_losses import material_consistency_loss


def stage1_loss(pred, target, gate):
    l_w1 = wasserstein_hist_loss(pred, target, 1.0 - gate)
    l_sfl = structural_fidelity_loss(pred, target)
    total = l_w1 + 0.1 * l_sfl
    return total, {"w1": l_w1, "sfl": l_sfl, "loss": total}


def stage2_loss(pred, target, ldr, gate):
    l_l2 = gated_l2_loss(pred, target, gate)
    l_sfl = structural_fidelity_loss(pred, target)
    l_mcl = material_consistency_loss(pred, ldr, gate)
    total = l_l2 + 0.5 * l_sfl + 0.3 * l_mcl
    return total, {"l2": l_l2, "sfl": l_sfl, "mcl": l_mcl, "loss": total}


def stage3_loss(pred, target, ldr, gate):
    l_l2 = gated_l2_loss(pred, target, gate)
    l_w1 = wasserstein_hist_loss(pred, target, 1.0 - gate)
    l_sfl = structural_fidelity_loss(pred, target)
    l_mcl = material_consistency_loss(pred, ldr, gate)
    l_sgcl = seam_gradient_continuity_loss(pred, target, gate)
    total = l_l2 + l_w1 + 0.2 * l_sfl + 0.1 * l_mcl + 0.3 * l_sgcl
    return total, {"l2": l_l2, "w1": l_w1, "sfl": l_sfl, "mcl": l_mcl, "sgcl": l_sgcl, "loss": total}


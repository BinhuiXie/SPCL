import torch.nn.functional as F
import torch.nn as nn


class PixelCLLoss(nn.Module):
    def __init__(self, cfg):
        super(PixelCLLoss, self).__init__()
        self.cfg = cfg

    def forward(self, semantic_prototype, src_feat, src_mask, tgt_feat, tgt_mask):
        """
        The proposed contrastive loss for class-wise alignment
        Args:
            semantic_prototype: (CxK) are source prototypes for K classes
            src_feat: (BxCxHxW) are source feature map
            src_mask: (BxHxW) are source mask
            tgt_feat: (BxCxHxW) are target feature map
            tgt_mask: (BxHxW) are target mask

        Returns:
        """
        assert not semantic_prototype.requires_grad

        # batch size, channel size, height and width of target sample
        _, _, Hs, Ws = src_feat.size()
        B, C, Ht, Wt = tgt_feat.size()
        # number of class
        K = semantic_prototype.size(1)

        # reshape src_feat to (BxHsxWs, C)
        src_feat = F.normalize(src_feat, p=2, dim=1)  # channel wise normalize
        src_feat = src_feat.transpose(1, 2).transpose(2, 3).contiguous()
        src_feat = src_feat.view(-1, C)

        # reshape tgt_feat to (BxHtxWt, C)
        tgt_feat = F.normalize(tgt_feat, p=2, dim=1)  # channel wise normalize
        tgt_feat = tgt_feat.transpose(1, 2).transpose(2, 3).contiguous()
        tgt_feat = tgt_feat.view(-1, C)

        src_mask = src_mask.view(-1, )
        tgt_mask = tgt_mask.view(-1, )

        # matrix-matrix product
        # (BHW, C) * (C, K) -> (BHW, K)
        src_dot_value = src_feat.mm(semantic_prototype) / 100.
        tgt_dot_value = tgt_feat.mm(semantic_prototype) / 100.

        ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
        loss = ce_criterion(src_dot_value, src_mask) + ce_criterion(tgt_dot_value, tgt_mask)

        return loss


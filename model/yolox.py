import torch.nn as nn
from .yolox_head import YOLOXHead
from .yolox_pafpn import YOLOPAFPN

class YOLOX(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = YOLOPAFPN()
        head = YOLOXHead(3)
        self.backbone = backbone
        self.head = head

    def forward(self, x,targets=None):
        fpn_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
        return outputs


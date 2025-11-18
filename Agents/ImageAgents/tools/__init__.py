"""Detection tools for image agents."""

from .nonescape_detect_tool import detect_with_nonescape_tool
#from .seresnet50_detect_tool import detect_with_seresnet50_tool
from .prithiv_v1_detect_tool import detect_with_prithiv_v1_tool
from .dima_vit_detect_tool import detect_with_dima_vit_tool
from .organika_sdxl_detect_tool import detect_with_organika_sdxl_tool
from .mobilenetv2_detect_tool import detect_with_mobilenetv2_tool

__all__ = [
    'detect_with_nonescape_tool',
    #'detect_with_seresnet50_tool',
    'detect_with_prithiv_v1_tool',
    'detect_with_dima_vit_tool',
    'detect_with_organika_sdxl_tool',
    'detect_with_mobilenetv2_tool',
]

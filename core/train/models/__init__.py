"""
Date: 2024-08-24 21:37:48
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 18:30:55
FilePath: /MAPS/core/train/models/__init__.py
"""

"""
only support fno_cnn for now
"""
from .factorfno_cnn import *
from .ffno_cnn import *
from .fno_cnn import *
from .layers import *

# from .simplecnn import *
from .neurolight_cnn import *
from .unet import *

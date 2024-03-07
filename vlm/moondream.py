import torch
from .visionencoder import visionencoder
from .configuration import config, phiconfig
from transformers import PreTrainedModel
import re
from .modelingphi import phiforcausalLM

class vlm(PreTrainedModel): 
  configclass = config

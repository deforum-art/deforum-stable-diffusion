import gc
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List

import clip
import torch
from helpers.aesthetics import load_aesthetics_model
from helpers.model_load import load_model_from_config, make_linear_decode
from helpers.render import (
    render_animation,
    render_image_batch,
    render_input_video,
    render_interpolation,
)
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

MODEL_CACHE = "diffusion_models_cache"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        pass

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        # inputs = request["inputs"]  # noqa
        # Invoke model and calculate predictions here.
        response["predictions"] = []
        return response

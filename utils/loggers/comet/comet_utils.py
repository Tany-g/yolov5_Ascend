"""# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================
"""

import logging
import os
from urllib.parse import urlparse

try:
    import comet_ml
except (ModuleNotFoundError, ImportError):
    comet_ml = None

import yaml

logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def download_model_checkpoint(opt, experiment):
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)

    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {model_name}")
        return

    model_asset_list = sorted(
        model_asset_list,
        key=lambda x: x["step"],
        reverse=True,
    )
    logged_checkpoint_map = {asset["fileName"]: asset["assetId"] for asset in model_asset_list}

    resource_url = urlparse(opt.weights)
    checkpoint_filename = resource_url.query

    if checkpoint_filename:
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME

    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment")
        return

    try:
        logger.info(f"COMET INFO: Downloading checkpoint {checkpoint_filename}")
        asset_filename = checkpoint_filename

        model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
        model_download_path = f"{model_dir}/{asset_filename}"
        with open(model_download_path, "wb") as f:
            f.write(model_binary)

        opt.weights = model_download_path

    except Exception as e:
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def set_opt_parameters(opt, experiment):
    """Update the opts Namespace with parameters
    from Comet's ExistingExperiment when resuming a run

    Args:
        opt (argparse.Namespace): Namespace of command line options
        experiment (comet_ml.APIExperiment): Comet API Experiment object
    """
    asset_list = experiment.get_asset_list()
    resume_string = opt.resume

    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            asset_id = asset["assetId"]
            asset_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string

    # Save hyperparameters to YAML file
    # Necessary to pass checks in training script
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    hyp_yaml_path = f"{save_dir}/hyp.yaml"
    with open(hyp_yaml_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path


def check_comet_weights(opt):
    """Downloads model weights from Comet and updates the
    weights path to point to saved weights location

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if weights are successfully downloaded
            else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.weights, str):
        if opt.weights.startswith(COMET_PREFIX):
            api = comet_ml.API()
            resource = urlparse(opt.weights)
            experiment_path = f"{resource.netloc}{resource.path}"
            experiment = api.get(experiment_path)
            download_model_checkpoint(opt, experiment)
            return True

    return None


def check_comet_resume(opt):
    """Restores run parameters to its original state based on the model checkpoint
    and logged Experiment parameters.

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if the run is restored successfully
            else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            api = comet_ml.API()
            resource = urlparse(opt.resume)
            experiment_path = f"{resource.netloc}{resource.path}"
            experiment = api.get(experiment_path)
            set_opt_parameters(opt, experiment)
            download_model_checkpoint(opt, experiment)

            return True

    return None

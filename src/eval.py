import pyrootutils
import time
import torch
import platform
import os
import cpuinfo  # Import cpuinfo for detailed CPU information
import psutil

root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "README.md" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

# Hack importing pandas here to bypass some conflicts with hydra
import pandas as pd

from typing import List, Tuple

import hydra
import torch
import torch_geometric
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
if not OmegaConf.has_resolver('eval'):
    OmegaConf.register_new_resolver('eval', eval)

log = utils.get_pylogger(__name__)


def log_system_info():
    """Logs the system's GPU and CPU information."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"Using GPU: {gpu_name} with {num_gpus} GPUs available")
    else:
        log.info("No GPU available, using CPU")

    num_cpus = len(os.sched_getaffinity(0))
    cpu_info = cpuinfo.get_cpu_info()
    cpu_model = cpu_info['brand_raw']
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cores_total = psutil.cpu_count()
    log.info(f"Using CPU: {cpu_model} with {num_cpus} physical CPUs and {physical_cores} physical cores.")
    log.info(f"Logical cores: {logical_cores}. Cores total: {cores_total}")

@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    
    log.info("Starting timing...")
    start_time = time.time() # Start timing
    log_system_info()  # Log the system info

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if float('.'.join(torch.__version__.split('.')[:2])) >= 2.0:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)


    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    
    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model, dynamic=True)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    end_time = time.time() # End timing
    log.info(f"Total inference time: {end_time - start_time:.2f} seconds")

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root + "/configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
# Last Change:  2024-05-01 11:29:15
import os
import argparse
import logging
import random
import typing

import torch
from mmf.common.registry import registry
from mmf.utils.build import build_config, build_trainer
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import distributed_init, get_rank, infer_init_method, is_xla
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.flags import flags
from mmf.utils.general import log_device_names
from mmf.utils.logger import setup_logger, setup_very_basic_config

logger = logging.getLogger(__name__)

setup_very_basic_config()


def main(configuration, init_distributed=False, predict=False):
    # A reload might be needed for imports
    setup_imports()
    configuration.import_user_dir()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    if init_distributed:
        distributed_init(config)

    seed = config.training.seed
    config.training.seed = set_seed(seed if seed == -1 else seed + get_rank())
    registry.register("seed", config.training.seed)

    config = build_config(configuration)

    setup_logger(
        color=config.training.colored_logs, disable=config.training.should_not_log
    )
    logger = logging.getLogger("mmf_cli.run")
    # Log args for debugging purposes
    logger.info(configuration.args)
    logger.info(f"Torch version: {torch.__version__}")
    log_device_names()
    logger.info(f"Using seed {config.training.seed}")

    trainer = build_trainer(config)
    trainer.load()
    if predict:
        trainer.inference()
    else:
        trainer.train()


def distributed_main(device_id, configuration, predict=False):
    config = configuration.get_config()
    config.device_id = device_id
    logger.info(f"Local rank of current node: {device_id}")
    main(configuration, init_distributed=True, predict=predict)


def run(opts: typing.Optional[typing.List[str]] = None, predict: bool = False):
    """Run starts a job based on the command passed from the command line.
    You can optionally run the mmf job programmatically by passing an optlist as opts.

    Args:
        opts (typing.Optional[typing.List[str]], optional): Optlist which can be used.
            to override opts programmatically. For e.g. if you pass
            opts = ["training.batch_size=64", "checkpoint.resume=True"], this will
            set the batch size to 64 and resume from the checkpoint if present.
            Defaults to None.
        predict (bool, optional): If predict is passed True, then the program runs in
            prediction mode. Defaults to False.
    """
    setup_imports()

    if opts is None:  # True
        parser = flags.get_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    configuration = Configuration(args)  # args is the command line script
    # Do set runtime args which can be changed by MMF
    configuration.args = args
    config = configuration.get_config()
    with open("config.txt", "w") as f:
        print(config, file=f)
    config.start_rank = 0
    if config.distributed.init_method is None:  # True
        infer_init_method(config)  
    
    if config.distributed.debug:
        config.device_id = 0
        main(configuration, predict=predict)
    else:
        logger.info(">>> Multi-Node Multi-GPU Training Forwarding")
        # Get the local rank of the current node
        local_rank = int(os.environ['LOCAL_RANK'])
        distributed_main(
            local_rank, 
            configuration, 
            predict,
        )

if __name__ == "__main__":
    run()

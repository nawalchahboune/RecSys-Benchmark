# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
from collections.abc import MutableMapping
from logging import getLogger
import os
import struct
import numpy as np
import torch

import torch.distributed as dist

from ray import tune

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    get_environment,
    get_flops,
    get_model,
    get_trainer,
    init_logger,
    init_seed,
    set_color,
)


import os, struct
import numpy as np
import torch
from logging import getLogger

def evaluate_and_export(trainer, test_data, config, saved, eval_only, train_data, model):
    test_result = trainer.evaluate(
        test_data,
        load_best_model=(saved and not eval_only),
        show_progress=config["show_progress"],
    )
    logger = getLogger()
    # logger.info(set_color("test result", "yellow") + f": {test_result}")

    save_path = config["save_submission_path"] if "save_submission_path" in config else os.environ.get("SAVE_SUBMISSION_PATH")
    if not save_path:
        return test_result

    K = int(os.environ.get("SUBMISSION_K", 0)) or int(config["topk"][0] if ("topk" in config and config["topk"]) else 10)

    # Top-K par session, avec dépaquetage du batch -> interaction
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
            scores = (
                model.full_sort_predict(interaction)
                if hasattr(model, "full_sort_predict")
                else model.predict(interaction)
            )
            if isinstance(scores, (tuple, list)):
                scores = scores[0]
            topk_idx = torch.topk(scores, K, dim=1).indices.cpu().numpy()
            preds.append(topk_idx)

    topk_matrix = np.vstack(preds) if preds else np.empty((0, K), dtype=np.int64)
    num_sessions = int(topk_matrix.shape[0])

    ds = train_data._dataset
    # id2tok = ds.id2token("item_id")
    mapping_path = config["clueweb_mapping_path"] if "clueweb_mapping_path" in config else os.environ.get("CLUEWEB_MAPPING_PATH")
    # Convert RecBole item ids -> tokens for mapping
    rid_flat = topk_matrix.ravel()
    tokens_flat = np.array(ds.id2token("item_id", rid_flat))
    tokens = tokens_flat.reshape(topk_matrix.shape)
    if mapping_path and os.path.isfile(mapping_path):
        token_to_internal = {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    token_to_internal[parts[0]] = int(parts[1])
        internal_ids = np.vectorize(lambda tok: token_to_internal.get(str(tok), 0))(tokens).astype(np.int32)
    else:
        # tokens sont des entiers (ex: ml-100k); force 0-index si nécessaire
        try:
            internal_ids = tokens.astype(np.int64)
        except Exception:
            internal_ids = np.vectorize(lambda t: int(str(t)) if str(t).isdigit() else 0)(tokens).astype(np.int64)
        if internal_ids.size and internal_ids.min() > 0:
            internal_ids = internal_ids - 1
        internal_ids = internal_ids.astype(np.int32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(struct.pack("<i", num_sessions))
        f.write(struct.pack("<i", K))
        f.write(internal_ids.ravel().tobytes())
    logger.info(set_color("submission saved", "yellow") + f": {save_path}")

    return test_result

def run(
    model,
    dataset,
    exp_name,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            exp_name=exp_name,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, exp_name, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_recbole(
    model=None,
    dataset=None,
    exp_name=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        exp_name=exp_name,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    if config["resume_path"]:
        logger.info(f"Loading model checkpoint from {config['resume_path']}")
        trainer.resume_checkpoint(config["resume_path"])

    # model training
    # best_valid_score, best_valid_result = trainer.fit(
    #     train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    # )
    # if not config.get("eval_only", False):
    #     best_valid_score, best_valid_result = trainer.fit(
    #         train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    #     )
    # else:
    #     logger.info("Eval-only: skipping training, using resumed checkpoint.")
    #     best_valid_score, best_valid_result = None, None

    # model evaluation
    # test_result = trainer.evaluate(
    #     test_data, load_best_model=saved, show_progress=config["show_progress"]
    # )
    
    # eval_only flag (Config n'a pas .get)
    eval_only = config["eval_only"] if "eval_only" in config else False

    # model training (skipped in eval_only mode)
    if not eval_only:
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )
    else:
        logger.info("Eval-only: skipping training, using resumed checkpoint.")
        best_valid_score, best_valid_result = None, None


    test_result = trainer.evaluate(
        test_data,
        load_best_model=(saved and not eval_only),
        show_progress=config["show_progress"],
    )
    
    # test_result = evaluate_and_export(trainer, test_data, config, saved, eval_only, train_data, model)

    # tune.report(**test_result)
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    # logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    # logger.info(set_color("test result", "yellow") + f": {test_result}")
    if best_valid_result is not None:
        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    else:
        logger.info(set_color("best valid ", "yellow") + ": skipped")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        # dist.destroy_process_group()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:4],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    # test_result = trainer.evaluate(test_data, load_best_model=saved)
    eval_only = config["eval_only"] if "eval_only" in config else False
    test_result = trainer.evaluate(
        test_data,
        load_best_model=(saved and not eval_only),
        show_progress=config["show_progress"],
    )
    # test_result = evaluate_and_export(trainer, test_data, config, saved, eval_only, train_data, model)
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    try:
        checkpoint = torch.load(model_file, weights_only=False)
    except TypeError:  # PyTorch<2.6
        checkpoint = torch.load(model_file)

    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

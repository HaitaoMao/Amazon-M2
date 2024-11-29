import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from SRGNNF import SRGNNF
import torch
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="GRU4Rec",
        help="Model for session-based rec.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="diginetica-session",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Whether evaluating on validation set (split from train set), otherwise on test set.",
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.1, help="ratio of validation set."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu id."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1024, help="train batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=256, help="test batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate."
    )
    parser.add_argument(
        "--dropout", type=float, default=0, help="dropout."
    )
    parser.add_argument(
        "--dropouts", type=str, default=0, help="dropout."
    )
    parser.add_argument(
        "--attn_dropout", type=float, default=0, help="dropout."
    )
    parser.add_argument(
        "--hidden_dropout", type=float, default=0,
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="num layers."
    )
    parser.add_argument(
        "--step", type=int, default=1, help="num layers."
    )
    parser.add_argument(
        "--saved_model", type=str, default=None, help="saved model."
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()
    # configurations initialization
    config_dict = {
        "data_path": "./",
        "dataset": args.dataset,
        "USER_ID_FIELD": "session_id",
        "load_col": None,
        "neg_sampling": None,
        "benchmark_filename": ["train", "test"],
        "alias_of_item_id": ["item_id_list"],
        "topk": [100],
        "metrics": ["Recall", "MRR"],
        "valid_metric": "MRR@100",
        'loss_type': 'CE',
        'train_neg_sample_args': None,
        "gpu_id": args.gpu,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "embedding_size": 64,
        "hidden_size": 128,
        "learning_rate": args.lr,
        "num_layers": args.num_layers,
        "dropout_prob": args.dropout,
        "dropout_probs": args.dropouts,
        "attn_dropout_prob": args.attn_dropout,
        "hidden_dropout_prob": args.hidden_dropout,
        "step": args.step,
        "selected_features": ["class"] if args.model.endswith('F') else [],
        "load_col": {'inter': ['session_id', 'item_id_list', 'item_id', 'item_locale'], 'item': ['item_id', 'embedding']} if args.model.endswith('F') else None,
        "pooling_mode": "sum",
        "numerical_features": ["embedding"] if args.model.endswith('F') else [],
    }
    # import ipdb; ipdb.set_trace()
    config = Config(
        model=args.model, dataset=f"{args.dataset}", config_dict=config_dict
    )
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    # train_dataset = dataset.build()[0]
    train_dataset, test_dataset = dataset.build()
    # import ipdb; ipdb.set_trace()
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio(
            [1 - args.valid_portion, args.valid_portion]
        )
        import ipdb; ipdb.set_trace()
        train_data = get_dataloader(config, "train")(
            config, new_train_dataset, None, shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, new_test_dataset, None, shuffle=False
        )
    else:
        train_data = get_dataloader(config, "train")(
            config, train_dataset, None, shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, None, shuffle=False
        )

    # model loading and initialization
    if args.model in ['GRU4Rec', 'GRU4RecF', 'NARM', 'SRGNN', 'STAMP', 'CORE']:
        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    else:
        model = globals()[args.model](config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.resume_checkpoint(f'saved/{args.saved_model}')
    # import ipdb; ipdb.set_trace()
    # model training and evaluation
    # test_score, test_result = trainer.fit(
    #     train_data, test_data, saved=True, show_progress=config["show_progress"]
    # )
    # logger.info(set_color("test result", "yellow") + f": {test_result}")
    # import ipdb; ipdb.set_trace()
    id2token = dataset.field2id_token['item_id_list']
    model.eval()
    all_indices = []
    for batch_idx, batched_data in enumerate(test_data):
        interaction, history_index, positive_u, positive_i = batched_data
        interaction = interaction.to(config['device'])
        scores = model.full_sort_predict(interaction)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        values, indices = scores.topk(100)
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0)
    predictions = dataset.field2id_token['item_id_list'][all_indices]
    df_test = pd.read_csv(f'./{args.dataset}/{args.dataset}.test.inter', sep='\t')
    predictions = predictions.tolist()
    df_test['next_item_prediction'] = predictions
    df_test = df_test.drop(df_test.index[:327049])
    # df_test = df_test.drop(columns=['Unnamed: 0'])
    df_test.to_csv(f'{args.dataset}/{args.model}_{args.dataset}_all.csv', sep='\t')
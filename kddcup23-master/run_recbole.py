import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='task1_uk', help='Datasets for session-based rec.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = parse_args()

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        'loss_type': 'BPR',
        'benchmark_filename': ['train', 'valid', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [5],
        'metrics': ['Recall', 'MRR'],
        'valid_metric': 'MRR@5',
    }

    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets

    # sampler
    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset, built_datasets
    )

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    valid_data = get_dataloader(config, 'valid')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(train_data, test_data, saved=True, show_progress=config['show_progress'])

    logger.info(set_color('test result', 'yellow') + f': {test_result}')

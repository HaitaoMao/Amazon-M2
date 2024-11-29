import os
import argparse
from pandas import read_parquet
from recbole.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='sample 3/task1/uk', help='Input file path of the original dataset in Parquet.')
    parser.add_argument('--output_path', '-o', type=str, default='dataset/task1_uk', help='Output path of atomic files required by RecBole.')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Ratio of training data to be valid data.')
    # parser.add_argument('')
    args = parser.parse_args()
    return args


def load_ori_train_dataset(input_path):
    pd_data = read_parquet(os.path.join(input_path, 'sessions_train.parquet'))
    np_data = pd_data.to_numpy()
    return np_data


def train_valid_split(all_data, val_ratio):
    n_data = len(all_data)
    n_valid = int(val_ratio * n_data)
    return all_data[n_valid:], all_data[:n_valid]


def load_ori_test_dataset(input_path):
    test = read_parquet(os.path.join(input_path, 'sessions_test.parquet'))
    gt = read_parquet(os.path.join(input_path, 'gt_test.parquet'))
    test.insert(1, 'next_item', gt['next_item'])
    return test.to_numpy()


def save_atomic_files(all_data, output_path):
    dataset_name = os.path.split(output_path)[-1]
    suffix_list = ['train', 'valid', 'test']
    session_id = 0

    for data, suffix in zip(all_data, suffix_list):
        output_filename = os.path.join(output_path, f'{dataset_name}.{suffix}.inter')
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write('session_id:token\titem_id_list:token_seq\titem_id:token\n')
            for prev_item, next_item, location in data:
                session_id += 1
                prev_item = ' '.join(map(str, prev_item.tolist()))
                file.write(f'{session_id}\t{prev_item}\t{next_item}\n')


if __name__ == '__main__':
    args = parse_args()

    assert os.path.exists(args.input_path), f'Input path [{args.input_path}] not exists.'
    ensure_dir(args.output_path)

    sessions_train_all = load_ori_train_dataset(args.input_path)
    sessions_train, sessions_valid = train_valid_split(sessions_train_all, args.valid_ratio)
    sessions_test = load_ori_test_dataset(args.input_path)
    save_atomic_files([sessions_train, sessions_valid, sessions_test], args.output_path)

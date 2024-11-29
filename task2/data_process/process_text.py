import argparse
import html
import os
import random
import re
import torch
from pandas import read_parquet
from tqdm import tqdm
from recbole.utils import ensure_dir

from utils import set_device, load_plm


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def generate_text(args, features):
    item_text_list = []

    raw_meta_pd_data = read_parquet(os.path.join(args.input_path, f'products.parquet'))
    raw_meta_data = raw_meta_pd_data.to_dict()
    for key in tqdm(raw_meta_data['id'].keys(), desc='Generate text'):
        text = ''
        for meta_key in features:
            if meta_key in raw_meta_data:
                meta_value = clean_text(raw_meta_data[meta_key][key])
                text += meta_value + ' '
        item_text_list.append([key, text])
    return item_text_list


def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(str(item) + '\t' + text + '\n')


def preprocess_text(args, features=['title']):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    # load item text and clean
    item_text_list = generate_text(args, ['title'])
    item_text_list = []

    item2key = {}

    raw_meta_pd_data = read_parquet(os.path.join(args.input_path, f'products.parquet'))
    raw_meta_data = raw_meta_pd_data.to_dict()
    for key in tqdm(raw_meta_data['id'].keys(), desc='Generate text'):
        text = ''
        for meta_key in features:
            if meta_key in raw_meta_data:
                meta_value = clean_text(raw_meta_data[meta_key][key])
                text += meta_value + ' '
        item_text_list.append([raw_meta_data['id'][key], text])
        item2key[raw_meta_data['id'][key]] = key
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list, item2key


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(f'{unit}\t{unit2index[unit]}\n')


def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)

    embeddings = []
    start, batch_size = 0, 4
    while start < len(texts):
        sentences = list(texts[start: start + batch_size])
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='sample 3/task1/uk')
    parser.add_argument('--output_path', '-o', type=str, default='dataset/pretrain')
    parser.add_argument('--dataset', '-d', type=str, default='task1_uk')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load item text from raw meta data file
    item_text_list, item2key = preprocess_text(args)

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    ensure_dir(os.path.join(args.output_path, args.dataset))

    # generate PLM emb and save to file
    generate_item_embedding(args, item_text_list, 
                            plm_tokenizer, plm_model, word_drop_ratio=-1)

    # save useful data
    write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_index(item2key, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2key'))

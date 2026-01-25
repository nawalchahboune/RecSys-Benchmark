import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import sys
from recbole.data import data_preparation
from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed

def _write_interaction_feat(feat, dataset, output_file):
    """Generic writer for Interaction (item_feat/user_feat) using numpy() and token maps."""
    if feat is None:
        print(f"Skip {output_file}: no features")
        return
    data = dict(feat.numpy())
    if not data:
        print(f"Skip {output_file}: empty features")
        return

    cols = list(data.keys())
    types = dataset.field2type  # e.g., token, token_seq, float
    token_maps = dataset.field2id_token  # dict: field -> list of tokens

    # Header "field:type"
    with open(output_file, "w", encoding="utf-8") as f:
        header = [f"{c}:{types.get(c,'token')}" for c in cols]
        f.write("\t".join(header) + "\n")

        n = len(next(iter(data.values())))
        for i in range(n):
            row = []
            for c in cols:
                v = data[c][i]
                t = types.get(c, "token")
                if t == "token":
                    tm = token_maps.get(c)
                    if tm is not None and isinstance(v, (int, np.integer)) and 0 <= int(v) < len(tm):
                        row.append(str(tm[int(v)]))
                    else:
                        row.append(str(v))
                elif t == "token_seq":
                    tm = token_maps.get(c, [])
                    seq = v if isinstance(v, (list, tuple, np.ndarray)) else [v]
                    toks = []
                    for s in seq:
                        if isinstance(s, (int, np.integer)) and 0 <= int(s) < len(tm):
                            toks.append(tm[int(s)])
                    row.append(" ".join(toks))
                else:
                    row.append(str(v))
            f.write("\t".join(row) + "\n")
    print(f"Saved: {output_file}")

def export_ml_item_file(dataset, output_file):
    _write_interaction_feat(getattr(dataset, "item_feat", None), dataset, output_file)

def export_user_file(dataset, output_file):
    _write_interaction_feat(getattr(dataset, "user_feat", None), dataset, output_file)



def export_raw_format(dataset, inter_dict, output_file):
    """Export interaction data to a tab-separated file in raw format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        field_types = {}
        for field in inter_dict.keys():
            if field in dataset.field2type:
                field_types[field] = dataset.field2type[field]
            else:
                field_types[field] = 'token'
        
        header = [f"{field}:{field_types.get(field, 'token')}" for field in inter_dict.keys() 
                  if not field.endswith('_list') and field != 'item_length']
        f.write('\t'.join(header) + '\n')
        
        for i in tqdm(range(len(inter_dict['user_id'])), desc=f"Writing {os.path.basename(output_file)}"):
            row = []
            for field in header:
                field_name = field.split(':')[0]
                if field_name in inter_dict:
                    row.append(str(inter_dict[field_name][i]))
            f.write('\t'.join(row) + '\n')
    print(f"Saved: {output_file}")


def export_amazon_item_file(dataset, output_file):
    """Export Amazon-style item file with mapped token fields and raw float fields."""
    item_feat = dataset.item_feat.numpy()

    # Extract fields
    item_id = item_feat['item_id']
    title_id = item_feat['title']
    category_id = item_feat['categories']
    avg_rating = item_feat['average_rating']
    rating_num = item_feat['rating_number']

    # Token mappings
    title_map = dataset.field2id_token['title']
    category_map = dataset.field2id_token['categories']

    # Open file and write header
    with open(output_file, 'w', encoding='utf-8') as writer:
        writer.write('item_id:token\ttitle:token\tcategories:token\taverage_rating:float\trating_number:float\n')

        for i in range(len(item_id)):
            iid = int(item_id[i])  # transformed item_id
            tid = int(title_id[i])
            cid = int(category_id[i])

            title_str = title_map[tid] if tid < len(title_map) else '[UNK]'
            category_str = category_map[cid] if cid < len(category_map) else '[UNK]'

            writer.write(f"{iid}\t{title_str}\t{category_str}\t{avg_rating[i]:.3f}\t{rating_num[i]:.3f}\n")

    print(f"Finished writing: {output_file}")




def export_mind_item_file(dataset, output_file):
    """Export Mind-style item file."""
    item_np = dataset.item_feat.numpy()
    item_id = item_np['item_id']
    title_id = item_np['title']
    category_id = item_np['categories']
    abstract_id = item_np['abstract']

    title_map = dataset.field2id_token['title']
    category_map = dataset.field2id_token['categories']
    abstract_map = dataset.field2id_token['abstract']

    with open(output_file, 'w', encoding='utf-8') as writer:
        writer.write('item_id:token\ttitle:token\tcategories:token\tabstract:token\n')
        for i in range(len(item_id)):
            iid = int(item_id[i])
            tid = int(title_id[i]); cid = int(category_id[i]); aid = int(abstract_id[i])
            writer.write(f"{iid}\t{title_map[tid]}\t{category_map[cid]}\t{abstract_map[aid]}\n")
    print(f"Finished writing: {output_file}")



def export_dataset_raw_format(model, dataset_name, output_dir, config_file_list=None, config_dict=None):
    """Export dataset in raw RecBole format"""
    # config = Config(model=model, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)
    
        # avoid RecBole warning about unrelated CLI args
    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0]]
    # no negative sampling with CE
    override = {
        "train_neg_sample_args": None,
        "valid_neg_sample_args": None,
        "test_neg_sample_args": None,
    }
    if config_dict:
        override.update(config_dict)
    config = Config(
        model=model,
        dataset=dataset_name,
        config_file_list=config_file_list,
        config_dict=override,
    )
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    sys.argv = argv_backup
    init_seed(config['seed'], config['reproducibility'])
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("Dataset preparation done.")
    
    os.makedirs(output_dir, exist_ok=True)

    train_dict = dict(train_data.dataset.inter_feat.numpy())
    valid_dict = dict(valid_data.dataset.inter_feat.numpy())
    test_dict = dict(test_data.dataset.inter_feat.numpy())
    
    export_raw_format(dataset, train_dict, os.path.join(output_dir, 'train.inter'))
    export_raw_format(dataset, valid_dict, os.path.join(output_dir, 'valid.inter'))
    export_raw_format(dataset, test_dict, os.path.join(output_dir, 'test.inter'))

    if 'amzn' in dataset_name:
        export_amazon_item_file(dataset, os.path.join(output_dir, 'item'))
    elif 'mind' in dataset_name: 
        export_mind_item_file(dataset, os.path.join(output_dir, 'item'))
    elif 'ml' in dataset_name: 
        export_ml_item_file(dataset, os.path.join(output_dir, 'item'))
    else: 
        raise NotImplementedError(f"{dataset_name}-specific export module is not implemented") 
        
    export_user_file(dataset, os.path.join(output_dir, 'user'))
    
    print("All files exported.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. NeuMF)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g. ml-1m)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--config_file_list', nargs='+', required=True, help='List of config file paths')

    args = parser.parse_args()

    export_dataset_raw_format(
        model=args.model,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        config_file_list=args.config_file_list
    )


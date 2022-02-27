"""Get dataset of (entity, occupation) tuples.

We source this from Wikidata, and then restrict to a set of entities which the
model has seen in its training data.
"""
import argparse
import json
import pathlib

from src.utils import env

import datasets
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dl and preprocess (entity, occ) data')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='link data here (default: project data dir)')
    args = parser.parse_args()

    data_dir = args.data_dir or env.data_dir()
    data_dir.mkdir(exist_ok=True, parents=True)

    # TODO(evandez): Just auto download TaskBench if needed?
    tb_root = data_dir / 'TaskBenchData/atomic'
    if not tb_root.exists():
        raise FileNotFoundError(
            'TaskBench not found; '
            'please download and unzip to $KF_DATA_DIR/TaskBench')

    tb_raw = []
    for tb_wikidata_dir in tb_root.glob('wiki{occupation(0)}'):
        file = tb_wikidata_dir / 'all.jsonl'
        with file.open('r') as handle:
            for line in tqdm(handle.readlines(), desc=file.parent.name):
                tb_raw.append(json.loads(line))

    # Keep only entities containing a wikipedia article, so that most large LMs
    # will have seen them during training.
    wikipedia = datasets.load_dataset('wikipedia', '20200501.en')
    assert isinstance(wikipedia, datasets.dataset_dict.DatasetDict), wikipedia

    articles_by_title = {}
    for sample in tqdm(wikipedia['train'], desc='index wikipedia articles'):
        title = sample['title']
        article = sample['text']
        articles_by_title[title.lower()] = article

    results, seen = [], set()
    for entry in tqdm(tb_raw, desc='select (entity, occupation) pairs'):
        entity = entry['inputs'][0]['ent_name'].lower()
        occupation = entry['train_tgts'][0]['ent_name'].lower()
        if entity in articles_by_title and entity not in seen:
            results.append({'entity': entity, 'occupation': occupation})
            seen.add(entity)

    data_file = data_dir / 'occupations.json'
    with data_file.open('w') as handle:
        json.dump(results, handle)

"""Get dataset of (entity, occupation) tuples.

We source this from Wikidata, and then restrict to a set of entities which the
model has seen in its training data.
"""
import argparse
import json
import pathlib

from src import wikidata
from src.utils import env

BANNED_OCCUPATIONS = {
    'pensioner',
}

SUPER_OCCUPATIONS = {
    'actor': {
        'film actor',
        'television actor',
        'television presenter',
    },
    'athlete': {
        'american football player',
        'australian rules football player',
        'association football player',
        'athletics competitor',
        'basketball player',
        'boxer',
        'cricketer',
        'ice hockey player',
        'rugby union player',
        'rower',
        'sport cyclist',
        'swimmer',
        'volleyball player',
    },
    'businessperson': {'merchant'},
    'doctor': {'physician'},
    'educator': {
        'pedagogue',
        'teacher',
        'university teacher',
    },
    'filmmaker': {'film director'},
    'historian': {'art historian'},
    'member of the military': {
        'military officer',
        'military personnel',
    },
    'musician': {
        'conductor',
        'guitarist',
        'singer-songwriter',
        'opera singer',
    },
    'politician': {'diplomat'},
    'religious figure': {
        'catholic priest',
        'pastor',
        'priest',
    },
    'scientist': {'researcher'},
    'sports manager': {'association football manager'},
    'writer': {
        'author',
        'novelist',
        'poet',
        'screenwriter',
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dl and preprocess (entity, occupation) data')
    parser.add_argument('--limit-occupations',
                        type=int,
                        default=100,
                        help='max occupations to query for')
    parser.add_argument('--limit-entities-per-occupation',
                        type=int,
                        default=100,
                        help='max entities to query per occupation')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='write data here (default: project data dir)')
    args = parser.parse_args()

    data_dir = args.data_dir or env.data_dir()
    data_dir.mkdir(exist_ok=True, parents=True)

    occupation_ids = wikidata.get_occupation_ids(limit=args.limit_occupations)

    occupation_entities = wikidata.get_occupations(occupation_ids)
    occupation_entities_by_id = {
        entity.entity_id: entity for entity in occupation_entities
    }

    people_entities_by_occupation = wikidata.get_entities_by_occupation(
        occupation_ids, limit=args.limit_entities_per_occupation)

    entries = []
    for occupation_id, people in people_entities_by_occupation.items():
        occupation = occupation_entities_by_id[occupation_id].get_label()
        if occupation in BANNED_OCCUPATIONS:
            continue

        for person in people:
            for key, superset in SUPER_OCCUPATIONS.items():
                if occupation.lower() in superset:
                    occupation = key
                    break
            entry = {'entity': person, 'occupation': occupation}
            entries.append(entry)

    occupations_file = data_dir / 'occupations.json'
    with occupations_file.open('w') as handle:
        json.dump(entries, handle)

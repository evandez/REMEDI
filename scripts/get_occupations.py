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
    'manager',  # Overly vague
    'official',  # Overly vague
    'pensioner',  # Too specific
    'pornographic actor',  # Inappropriate
    'trade unionist',  # Too specific
}

SUPER_OCCUPATIONS = {
    'academic': {
        'literary scholar',
        'professor',
    },
    'actor': {
        'film actor',
        'stage actor',
        'television actor',
        'television presenter',
        'performing artist',
        'seiyū',
        'voice actor',
    },
    'pilot': {'aircraft pilot'},
    'athlete': {
        'alpine skier',
        'amateur wrestler',
        'american football player',
        'australian rules football player',
        'artistic gymnast',
        'association football player',
        'athletics competitor',
        'badminton player',
        'baseball player',
        'basketball player',
        'boxer',
        'canoeist',
        'cricketer',
        'figure skater',
        'field hockey player',
        'golfer',
        'handball player',
        'ice hockey player',
        'rugby league player',
        'rugby union player',
        'rower',
        'sport cyclist',
        'sport shooter',
        'sprinter',
        'swimmer',
        'volleyball player',
    },
    'businessperson': {
        'merchant',
        'business executive',
    },
    'critic': {'literary critic'},
    'doctor': {
        'physician',
        'surgeon',
    },
    'editor': {
        'contributing editor',
        'film editor',
    },
    'educator': {
        'high school teacher',
        'music pedagogue',
        'pedagogue',
        'teacher',
        'university teacher',
    },
    'entrepreneur': {'inventor'},
    'film artist': {
        'cinematographer',
        'scenographer',
    },
    'director': {'film director', 'theatrical director'},
    'historian': {
        'archivist',
        'art historian',
    },
    'journalist': {'opinion journalist'},
    'lawyer': {'barrister'},
    'linguist': {
        'esperantist',
        'classical philologist',
        'philologist',
    },
    'publisher': {'printmaker', 'printer'},
    'member of the military': {
        'military commander',
        'military officer',
        'military leader',
        'military personnel',
        'naval officer',
        'soldier',
    },
    'musician': {
        'conductor',
        'disc jockey',
        'guitarist',
        'jazz musician',
        'opera singer',
        'organist',
        'lyricist',
        'pianist',
        'rapper',
        'singer-songwriter',
        'songwriter',
        'violinist',
    },
    'politician': {
        'civil servant',
        'diplomat',
        'political candidate',
        'statesperson',
    },
    'religious figure': {
        'catholic priest',
        'christian minister',
        'pastor',
        'priest',
        'rabbi',
    },
    'producer': {
        'film producer',
        'record producer',
        'television producer',
    },
    'racecar driver': {'racing automobile driver'},
    'referee': {'association football referee'},
    'scientist': {'researcher'},
    'sports manager': {'association football manager', 'basketball coach'},
    'visual artist': {
        'drawer',
        'cartoonist',
        'comics artist',
        'mangaka',
        'graphic artist',
        'graphic designer',
    },
    'writer': {
        'author',
        "children's writer",
        'essayist',
        'novelist',
        'non-fiction writer',
        'poet',
        'science fiction writer',
        'screenwriter',
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dl and preprocess (entity, occupation) data')
    parser.add_argument('--limit-occupations',
                        type=int,
                        default=200,
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

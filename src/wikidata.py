"""Functions for downloading data from Wikidata."""
from typing import Any, Mapping, Sequence

from qwikidata import entity, linked_data_interface, sparql
from retry import retry
from tqdm.auto import tqdm


def get_occupation_ids(limit: int = 100) -> Sequence[str]:
    """Return all occupation Wikidata IDs.

    Args:
        limit (int, optional): Maximum number of occupations to return.

    Returns:
        Sequence[str]: IDs of most common occupations, up to limit.

    """
    query = '''
SELECT ?occupation (COUNT(*) AS ?c)
WHERE {
    ?person wdt:P106 ?occupation.
}
GROUP BY ?occupation
ORDER BY DESC(?c)'''
    query += f'LIMIT {limit}'
    results = sparql.return_sparql_query_results(query)
    ids = [
        result['occupation']['value'].split('/')[-1]
        for result in results['results']['bindings']
    ]
    return tuple(ids)


def get_occupations(
        occupation_ids: Sequence[str]) -> Sequence[entity.WikidataItem]:
    """Get English names for occupations.

    Args:
        occupation_ids (Sequence[str]): The occupation IDs.

    Returns:
        Sequence[str]: The corresponding occupation names.

    """
    occupations = []
    for occupation_id in tqdm(occupation_ids, desc='get occupations'):
        params = linked_data_interface.get_entity_dict_from_api(occupation_id)
        occupation = entity.WikidataItem(params)
        occupations.append(occupation)
    return tuple(occupations)


@retry()
def get_entities_with_occupation(occupation_id: str,
                                 limit: int = 100) -> Sequence[str]:
    """Get the names of the most famous entities with the occupation.

    Fame is measured by site links.

    Args:
        occupation_id (str): The occupation ID.
        limit (int, optional): Maximum number of entities to return.

    Returns:
        Sequence[str]: Names of famous entities with that occupation.

    """
    query = '''
SELECT DISTINCT ?item ?itemLabel ?sitelinks
WHERE {
    ?item wdt:P31 wd:Q5;  # Any instance of a human.'''
    query += f'''
    wdt:P106 wd:{occupation_id};  # Has the target occupation.
    wdt:P27 wd:Q30;  # Has US citizenship.
    wikibase:sitelinks ?sitelinks.
'''
    query += '''
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en,nl" }
}
ORDER BY DESC(?sitelinks)'''
    query += f'LIMIT {limit}'
    results = sparql.return_sparql_query_results(query)
    names = [
        result['itemLabel']['value']
        for result in results['results']['bindings']
    ]
    return tuple(names)


def get_entities_by_occupation(occupation_ids: Sequence[str],
                               **kwargs: Any) -> Mapping[str, Sequence[str]]:
    """Map occupation ID to entities with that occupation.

    The **kwargs are forwarded to get_entities_with_occupation.

    Args:
        occupation_ids (Sequence[str]): The occupation IDs.

    Returns:
        Mapping[str, Sequence[str]]: Mapping from occupation to entity names.

    """
    entities_by_occupation = {}
    for occupation_id in tqdm(occupation_ids,
                              desc='get entities for each occupation'):
        entities = get_entities_with_occupation(occupation_id, **kwargs)
        entities_by_occupation[occupation_id] = entities
    return entities_by_occupation

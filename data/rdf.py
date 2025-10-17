#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from jsonargparse import CLI
from jsonargparse.typing import register_type
import json

from rdflib import Graph


register_type(Path, type_check=lambda v, t: isinstance(v, t))


def parse(root: Path):
    g = Graph()
    for path in tqdm(list(root.glob("*.ttl"))):
        g.parse(path)
        print(path.name, len(g))
    return g


def verbose_query(g, query):
    print(f"Querying...\n{query}\n")
    result = g.query(query)
    print(f"Got {len(result)} results")
    return result


def multiparallel(g):
    """
    Main query for multiparallel data.
    Retrieves, for a given speech:
        - all of its translations
        - its source language
        - speaker identifier
        - for each type of party:
            - party identifier
        - national party identifier
        - its date
    """
    query="""
    SELECT ?text ?translation ?speech ?party ?partytypelabel ?date ?speaker
    WHERE { 
        ?speech lpv:spokenText ?text.
        ?speech dcterms:isPartOf ?agenda.
        ?agenda dcterms:date ?date.	
        ?speech lpv:spokenAs ?function.
        ?function lpv:institution ?party.
        ?party rdf:type ?partytype.
        ?partytype rdfs:label ?partytypelabel.
        ?speech lpv:speaker ?speaker.
        ?speech lpv:translatedText ?translation.
    }
    """
    df = {}
    result = verbose_query(g, query)
    for row in result:
        df.setdefault(row.speech, {})
        df[row.speech][row.text.language] = row.text.value
        df[row.speech][row.translation.language] = row.translation.value
        df[row.speech]["src_lang"] = row.text.language
        df[row.speech]["date"] = str(row.date.value)
        df[row.speech]["speaker"] = str(row.speaker)
        df[row.speech][row.partytypelabel] = str(row.party)

    df = pd.DataFrame(df).T
    print(len(df), df.columns)
    return df


def query_speakers(g):
    query = """
    SELECT ?speaker ?name ?dob ?country ?countryLabel
    WHERE { 
        ?speaker foaf:name ?name.
        ?speaker lpv:dateOfBirth ?dob.
        ?speaker lpv:countryOfRepresentation ?country.
        ?country rdfs:label ?countryLabel.
    }
    """
    speakers = {}
    result = verbose_query(g, query)
    for row in result:
        speaker = str(row.speaker)
        speakers[speaker] = {}
        speakers[speaker]["name"] = row.name.value
        speakers[speaker]["dateOfBirth"] = str(row.dob.value)
        speakers[speaker]["countryOfRepresentation"] = str(row.country)
        speakers[speaker]["countryOfRepresentationLabel"] = row.countryLabel.value
    print(len(speakers))
    return speakers


def query_parties(g):
    query = """
    SELECT ?party ?partylabel ?partytypelabel ?partyAcronym
    WHERE { 
        ?party rdf:type ?partytype.
        ?party rdfs:label ?partylabel.
        ?partytype rdfs:label ?partytypelabel.
        ?party lpv:acronym ?partyAcronym.
    }
    """
    parties = {}
    result = verbose_query(g, query)
    for row in result:
        party = str(row.party)
        parties[party] = {}
        parties[party]["label"] = row.partylabel.value
        parties[party]["type"] = row.partytypelabel.value
        parties[party]["acronym"] = row.partyAcronym.value
    return parties


def main(root: Path):
    g = parse(root)
    df = multiparallel(g)
    df.to_csv(root/"multi-europarl.csv")
    speakers = query_speakers(g)
    with open(root/"speakers.json", "wt") as file:
        json.dump(speakers, file)
    parties = query_parties(g)
    with open(root/"parties.json", "wt") as file:
        json.dump(parties, file)

if __name__ == "__main__":
    CLI(main)
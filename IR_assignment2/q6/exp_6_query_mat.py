
import re
from collections import defaultdict
from ahocorasick import Automaton 


def convert_queries_to_1d_array(file_path):
    queries_array = []
    q_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query = parts[1]
                queries_array.append(query)
                q_ids.append(parts[0])
    
    return queries_array,q_ids

with open('updated_entities.txt', 'r') as f:
    entities = [line.split('\t')[0].lower() for line in f.read().splitlines()]


automaton = Automaton()
for idx, entity in enumerate(entities):
    entity = " " + entity + " "
    automaton.add_word(entity, (idx, entity))
automaton.make_automaton()

queries, qids = convert_queries_to_1d_array('IR_assignment2/nfcorpus/dev.vid-desc.queries')

queries = [query.lower() for query in queries]
for q in queries:
    q = re.sub(r'[\.\n,\?\t\b]+', ' ', q)
    q= re.sub(r'[^a-zA-Z0-9\s]', '', q)


def find_entities_in_queries(queries, automaton):
    query_entity_map = defaultdict(list)
    for query,qid in zip(queries,qids):
        seen = set() 
        for end_index, (entity_idx, padded_entity) in automaton.iter(query):
            entity = padded_entity.strip()
            if entity not in seen: 
                seen.add(entity)
                query_entity_map[qid].append(entity)
    return query_entity_map

query_entity_map = find_entities_in_queries(queries, automaton)

with open('query_entity_exp6.txt', 'w', encoding='utf-8') as file:
    for query, entities in query_entity_map.items():
        row_text = '\t'.join([query] + entities)
        file.write(row_text + '\n')


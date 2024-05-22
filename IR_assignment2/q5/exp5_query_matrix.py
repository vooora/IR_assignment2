
import re
from collections import defaultdict
from ahocorasick import Automaton  

doc_dump_path = 'raw/doc_dump.txt'
docs = []
docids = []


with open(doc_dump_path, 'r', encoding='utf-8') as file:
    
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            doc_id, url, title, abstract = parts[0], parts[1], parts[2], parts[3]
            full_text = url + " " + title + " " + abstract 
            full_text = re.sub(r'[\.\n,\?\t\b]+', ' ', full_text)  
            full_text = re.sub(r'[^a-zA-Z0-9\s]', '', full_text)
            docs.append(full_text)
            docids.append(doc_id)
            

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

with open('knowledge_graphs/entities.txt', 'r') as f:
    entities = f.read().splitlines()
entities = [entity.lower() for entity in entities]

automaton = Automaton()
for idx, entity in enumerate(entities):
    entity = " " + entity + " "
    automaton.add_word(entity, (idx, entity))
automaton.make_automaton()

queries, qids = convert_queries_to_1d_array('train.nontopic-titles.queries')

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


with open('query_entity_mappings.txt', 'w', encoding='utf-8') as file:
    for query, entities in query_entity_map.items():
        
        row_text = '\t'.join([query] + entities)
        file.write(row_text + '\n')


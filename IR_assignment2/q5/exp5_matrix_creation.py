import csv
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
    
    return queries_array, q_ids

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


with open('knowledge_graphs/entities.txt', 'r') as f:
    entities = f.read().splitlines()
entities = [entity.lower() for entity in entities]

matrix = defaultdict(list)

automaton = Automaton()
for idx, entity in enumerate(entities):
    entity = " " + entity + " "
    automaton.add_word(entity, (idx, entity))
automaton.make_automaton()

for doc_idx, doc in enumerate(docs):
    for end_index, (entity_idx, padded_entity) in automaton.iter(doc):
        matrix[entities[entity_idx]].append(doc_idx+1)  



with open('query_entity_matrix.txt', 'w', encoding='utf-8') as file:
    
    for entity, row in matrix.items():
        row_text = '\t'.join([entity] + ["MED-" + str(r) for r in row])
        file.write(row_text + '\n')


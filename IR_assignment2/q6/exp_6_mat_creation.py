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

docs = []
docids = []
doc_dump_path = 'IR_assignment2/nfcorpus/raw/doc_dump.txt'
with open(doc_dump_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            doc_id, url, title, abstract = parts[0], parts[1], parts[2], parts[3]
            full_text = " ".join([url, title, abstract])
            full_text = re.sub(r'[\.\n,\?\t\b]+', ' ', full_text)  
            full_text = re.sub(r'[^a-zA-Z0-9\s]', '', full_text)
            docs.append(full_text)
            docids.append(doc_id)

entities_scores = {}
with open('updated_entities.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity, score = parts[0], float(parts[1])
            entities_scores[entity] = score


matrix = defaultdict(list)

automaton = Automaton()
for idx, entity in enumerate(entities_scores):
    padded_entity = " " + entity + " "  
    automaton.add_word(padded_entity, (idx, padded_entity))
automaton.make_automaton()

for doc_idx, doc in enumerate(docs):
    for end_index, (entity_idx, padded_entity) in automaton.iter(doc):
        entity = padded_entity.strip()
        matrix[entity].append(doc_idx + 1) 

with open('exp6mat.txt', 'w', encoding='utf-8') as file:
    for entity, doc_list in matrix.items():
        score = entities_scores[entity]  
        row_text = '\t'.join([entity] + ["MED-" + str(doc) for doc in doc_list] + [str(score)])
        file.write(row_text + '\n')


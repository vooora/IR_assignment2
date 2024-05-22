import re
from collections import defaultdict

def load_entity_doc_map(filename):
    entity_doc_map = defaultdict(lambda: {'docs': [], 'score': 0})
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            entity = parts[0]
            documents = parts[1:-1]  
            score = float(parts[-1]) 
            entity_doc_map[entity] = {'docs': documents, 'score': score}
    return entity_doc_map

def load_query_entity_map(filename):
    query_entity_map = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query = parts[0]
            entities = parts[1:]
            query_entity_map[query] = entities
    return query_entity_map

def score_queries(entity_doc_map, query_entity_map):
    query_scores = defaultdict(dict) 
    for query, entities in query_entity_map.items():
        doc_scores = defaultdict(float)  
        for entity in entities:
            if entity in entity_doc_map:
                entity_info = entity_doc_map[entity]
                for doc in entity_info['docs']:
                    doc_scores[doc] += entity_info['score']
        query_scores[query] = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return query_scores


def write_scores_to_file(query_scores, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for query, scores in query_scores.items():
            
            for doc, score in scores:
                file.write(f'{query} Q0 {doc} 0 {score} STANDARD\n')
            file.write('\n')

def main():
    entity_doc_map = load_entity_doc_map('exp6mat.txt')
    query_entity_map = load_query_entity_map('query_entity_exp6.txt')
    query_scores = score_queries(entity_doc_map, query_entity_map)
    write_scores_to_file(query_scores, 'tests.txt')

if __name__ == "__main__":
    main()



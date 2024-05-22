def convert_queries_to_2d_array(file_path):
    queries_array = []
    q_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query = parts[1]
                words = query.split(' ')
                queries_array.append(words)
                q_ids.append(parts[0])
    return queries_array, q_ids

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

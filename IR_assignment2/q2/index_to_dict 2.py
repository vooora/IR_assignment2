def build_tf_index(file_path):
    tf_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                doc_id, term, freq = parts
                freq = int(freq)
                if doc_id not in tf_index:
                    tf_index[doc_id] = {}
                tf_index[doc_id][term] = freq
    return tf_index

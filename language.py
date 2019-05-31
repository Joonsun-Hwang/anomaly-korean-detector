import json
import collections
import numpy as np


class Language:
    def __init__(self, file_path_tokens_map, file_path_vectors_map):
        with open(file_path_tokens_map, 'r') as f:
            self.token_map = json.loads(f.read())

        self.vector_map = {}
        with open(file_path_vectors_map, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.vector_map[int(data[0])] = np.array(data[1:]).astype(float)
        self.vector_map = collections.OrderedDict(sorted(self.vector_map.items()))

    def get_n_tokens(self):
        return len(self.token_map)

    def get_vectors(self):
        return np.array(list(self.vector_map.values()))

    def get_vector(self, i):
        return self.vector_map[i]

    @staticmethod
    def save_vectors(file_path_vector_map, vectors):
        with open(file_path_vector_map, 'w') as f:
            for idx, vector in enumerate(vectors):
                vector_map = str(idx) + ' ' + ' '.join(np.array2string(
                    vector, formatter={'float_kind':lambda x: "%.8f" % x})[1:-1].split()) + '\n'
                f.write(vector_map)

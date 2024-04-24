import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from src.type_based.special_tokens import NO_TOKEN


def create_matrix(path, types_dictionary ):
    all_triplets = set()
    for item in jsonlines.open(path):
        for triplet in item["triplets"]:
            all_triplets.add((triplet["head_id"], triplet["label_id"], triplet["tail_id"]))

    property_dictionary = defaultdict(set)
    for s, p, o in all_triplets:
        if s.startswith("Q") and o.startswith("Q"):
            property_dictionary[p].add((s, o))

    all_types = set()
    types_counter = defaultdict(lambda: (Counter(), Counter()))
    for p in property_dictionary:
        for s, o in property_dictionary[p]:
            s_types = types_dictionary.get(s, [NO_TOKEN])
            o_types = types_dictionary.get(o, [NO_TOKEN])
            all_types.update(s_types)
            all_types.update(o_types)
            for s_type in s_types:
                types_counter[p][0][s_type] += 1
            for o_type in o_types:
                types_counter[p][1][o_type] += 1

    normalized_types_counter = {}
    for p in types_counter:
        normalized_types_counter[p] = (dict(types_counter[p][0]), dict(types_counter[p][1]))
        for s_type in normalized_types_counter[p][0]:
            normalized_types_counter[p][0][s_type] /= len(property_dictionary[p])
        for o_type in normalized_types_counter[p][1]:
            normalized_types_counter[p][1][o_type] /= len(property_dictionary[p])

    all_types = sorted(list(all_types))
    all_types_dict = {x: i for i, x in enumerate(all_types)}
    number_of_types = len(all_types)

    properties = list(normalized_types_counter.keys())
    subject_matrix = np.zeros((len(property_dictionary), number_of_types))
    object_matrix = np.zeros((len(property_dictionary), number_of_types))
    for i, p in enumerate(properties):
        for s_type in normalized_types_counter[p][0]:
            subject_matrix[i, all_types_dict[s_type]] = normalized_types_counter[p][0].get(s_type, 0.0)
        for o_type in normalized_types_counter[p][1]:
            object_matrix[i, all_types_dict[o_type]] = normalized_types_counter[p][1].get(o_type, 0.0)

    subject_matrix_dot_product = np.dot(subject_matrix, subject_matrix.T)
    object_matrix_dot_product = np.dot(object_matrix, object_matrix.T)
    cosine_similarity_subject = subject_matrix_dot_product / (
        np.linalg.norm(subject_matrix, axis=1, keepdims=True) * np.linalg.norm(subject_matrix, axis=1, keepdims=True).T
    )
    cosine_similarity_object = object_matrix_dot_product / (
        np.linalg.norm(object_matrix, axis=1, keepdims=True) * np.linalg.norm(object_matrix, axis=1, keepdims=True).T
    )

    combined = cosine_similarity_subject * cosine_similarity_object
    return combined, properties


if __name__ == "__main__":
    types_dictionary = pickle.load(open("entity_types_alt.pickle", "rb"))

    main_paths = ["wiki/unseen_5", "wiki/unseen_10", "wiki/unseen_15",
                  "fewrel/unseen_5", "fewrel/unseen_10", "fewrel/unseen_15"]
    for main_path in main_paths:
        paths = []
        for seed in [0,1,2,3,4]:
            paths.append(Path(f"{main_path}_seed_{seed}/dev_mapped.jsonl"))


        subplots = []
        lower_triangular_means = []
        maximum= []
        for path in paths:
            matrix, _ = create_matrix(path, types_dictionary)
            subplots.append(matrix)
            lower_triangular_means.append(np.mean(np.tril(matrix, k=-1)))
            maximum.append(np.max(np.tril(matrix, k=-1)))

        fig, axs = plt.subplots(1, 5)
        for i, subplot in enumerate(subplots):
            max_value = maximum[i]
            mean_value = lower_triangular_means[i]
            axs[i].matshow(subplot)
            axs[i].set_title(f"Seed {i}")
            axs[i].set_xlabel(f"Max: {max_value:.2f}\nMean: {mean_value:.2f}")


        plt.savefig(f"{main_path}_matrix.png")





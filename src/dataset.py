import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import jsonlines
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.analyse_type_distribution import create_matrix
from src.type_based.special_tokens import NO_TOKEN


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item


def custom_collate_fn(batch, tokenizer, alt_tokenizer=None):
    labels = []
    texts = []
    type_texts = []
    elements_per_item = []
    batch_relations = []
    all_indices = []
    head_type_indices = []
    tail_type_indices = []
    head_type_string_indices = []
    tail_type_string_indices = []
    head_entity_positions = []
    tail_entity_positions = []
    encountered_types = {}
    counter = 0
    for idx, item in enumerate(batch):
        labels = labels + item[1]
        all_indices.append(item[3])
        for elem in item[0]:
            texts.append(elem[0][0])
            head_entity_positions.append(elem[0][1])
            tail_entity_positions.append(elem[0][2])
            if elem[1] is not None:
                for type_ in elem[1][0]:
                    if type_ not in encountered_types:
                        encountered_types[type_] = len(encountered_types)
                        type_texts.append(type_)
                    head_type_indices.append(counter)
                    head_type_string_indices.append(encountered_types[type_])
                for type_ in elem[1][1]:
                    if type_ not in encountered_types:
                        encountered_types[type_] = len(encountered_types)
                        type_texts.append(type_)
                    tail_type_indices.append(counter)
                    tail_type_string_indices.append(encountered_types[type_])
            counter += 1

        batch_relations.append(item[2])
        elements_per_item.append(len(item[0]))
    maximum_elements = max(elements_per_item)
    assert all([x == maximum_elements for x in elements_per_item])
    head_token_positions = []
    tail_token_positions = []
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    for offset_mapping, head_entity_position, tail_entity_position in zip(encoded["offset_mapping"], head_entity_positions, tail_entity_positions):
        elem_head_token_positions = []
        elem_tail_token_positions = []
        for idx, offset in enumerate(offset_mapping):
            if offset[0] >= head_entity_position[0] and offset[1] <= head_entity_position[1]:
                elem_head_token_positions.append(idx)
            if offset[0] >= tail_entity_position[0] and offset[1] <= tail_entity_position[1]:
                elem_tail_token_positions.append(idx)
        head_token_positions.append(torch.tensor(elem_head_token_positions))
        tail_token_positions.append(torch.tensor(elem_tail_token_positions))
    assert len(head_token_positions) == len(tail_token_positions)
    head_token_positions = pad_sequence(head_token_positions, batch_first=True, padding_value=-1)
    tail_token_positions = pad_sequence(tail_token_positions, batch_first=True, padding_value=-1)
    labels = torch.tensor(labels, dtype=torch.float)
    head_type_indices = torch.tensor(head_type_indices, dtype=torch.long)
    tail_type_indices = torch.tensor(tail_type_indices, dtype=torch.long)
    head_type_string_indices = torch.tensor(head_type_string_indices, dtype=torch.long)
    tail_type_string_indices = torch.tensor(tail_type_string_indices, dtype=torch.long)
    if not type_texts:
        encoded_types = None
    else:
        encoded_types = tokenizer(type_texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)

    return encoded, encoded_types, labels, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, maximum_elements, batch_relations, all_indices

class RelationExtractionDataset(pl.LightningDataModule):
    def __init__(self, dataset_name: str, tokenizer, alt_tokenizer=None, args: dict = None):
        if args is None:
            args = {}
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.rel_to_description = {item["wikidata_id"]:item["en_description"] for item in jsonlines.open("rel_id_title_description_cleaned.jsonl")}

        self.type_descriptions = json.load(open("types_to_description_alt.json"))

        if Path("entity_types_reparsed.pickle").exists():
            self.types_dictionary = pickle.load(open("entity_types_reparsed.pickle", "rb"))
        else:
            self.types_dictionary = pickle.load(open("entity_types.pickle", "rb"))
            types_index = pickle.load(open("entity_types_index.pickle", "rb"))
            inverse_types_index = {v: k for k, v in types_index.items()}
            self.types_dictionary = {key: [inverse_types_index[x] for x in value] for key, value in self.types_dictionary.items()}
            pickle.dump(self.types_dictionary, open("entity_types_reparsed.pickle", "wb"))

        self.type_labels = json.load(open("types_to_label.json"))
        self.other_properties = args.get("other_properties", 5)
        self.hard_other_properties = args.get("hard_other_properties", 5)
        self.tokenizer = tokenizer
        self.alt_tokenizer = alt_tokenizer
        self.num_workers = args.get("num_workers", 2)
        self.include_descriptions = args.get("include_descriptions", False)
        self.include_types = args.get("include_types", False)
        self.use_predicted_candidates = args.get("use_predicted_candidates", False)


    def prepare_data(self):
        pass

    def create_representation(self, text, head_text, tail_text, property):
        output_text = f"[CLS] Given the Head Entity : {head_text}, Tail Entity : {tail_text} and Context : {text}, the context expresses the relation [SEP] {property[0]} [SEP]"
        head_text_start = output_text.find(f"Head Entity : {head_text}") + len(f"Head Entity : ")

        head_text_span = (head_text_start, head_text_start + len(head_text))
        tail_text_start = output_text.find(f"Tail Entity : {tail_text}") + len(f"Tail Entity : ")
        tail_text_span = (tail_text_start, tail_text_start + len(tail_text))
        assert output_text[head_text_span[0]:head_text_span[1]] == head_text
        assert output_text[tail_text_span[0]:tail_text_span[1]] == tail_text
        return output_text, head_text_span, tail_text_span

    def create_type_based_representation(self, head_text, tail_text, head_qid, tail_qid, property):
        head_types = self.types_dictionary.get(head_qid, [])
        tail_types = self.types_dictionary.get(tail_qid, [])
        head_type_descriptions = " and ".join([self.type_labels[x] for x in head_types if x in self.type_labels])
        tail_type_descriptions = " and ".join([self.type_labels[x] for x in tail_types if x in self.type_labels])
        if not head_type_descriptions:
            head_type_descriptions = NO_TOKEN
        if not tail_type_descriptions:
            tail_type_descriptions = NO_TOKEN
        property_description = self.rel_to_description[property[1]]
        return f"[CLS] Given the Head Types : {head_type_descriptions}, Tail Types : {tail_type_descriptions}, they match the relation description [SEP] {property_description} [SEP]"

    def create_type_representation(self, head_qid, tail_qid):
        head_types = self.types_dictionary.get(head_qid, [])
        tail_types = self.types_dictionary.get(tail_qid, [])
        head_type_strings= []
        for type_ in head_types:
            if type_ in self.type_labels:
                type_string = self.type_labels[type_]
                if type_ in self.type_descriptions:
                    type_string = f"{type_string} defined as {self.type_descriptions[type_]}"
                head_type_strings.append(f"[CLS] {type_string} [SEP]")
        if len(head_type_strings) == 0:
            head_type_strings = [f"[CLS] {NO_TOKEN} [SEP]"]
        tail_type_strings = []
        for type_ in tail_types:
            if type_ in self.type_labels:
                type_string = self.type_labels[type_]
                if type_ in self.type_descriptions:
                    type_string = f"{type_string} defined as {self.type_descriptions[type_]}"
                tail_type_strings.append(f"[CLS] {type_string} [SEP]")
        if len(tail_type_strings) == 0:
            tail_type_strings = [f"[CLS] {NO_TOKEN} [SEP]"]
        return head_type_strings, tail_type_strings

    def create_representation_with_description(self, text, head_text, tail_text, property):
        property_description = self.rel_to_description[property[1]]

        output_text = f"[CLS] Given the Head Entity : {head_text}, Tail Entity : {tail_text} and Context : {text}, the context expresses the relation [SEP] {property[0]} defined as {property_description} [SEP]"

        head_text_start = output_text.find(f"Head Entity : {head_text}") + len(f"Head Entity : ")

        head_text_span = (head_text_start, head_text_start + len(head_text))
        tail_text_start = output_text.find(f"Tail Entity : {tail_text}") + len(f"Tail Entity : ")
        tail_text_span = (tail_text_start, tail_text_start + len(tail_text))
        assert output_text[head_text_span[0]:head_text_span[1]] == head_text
        assert output_text[tail_text_span[0]:tail_text_span[1]] == tail_text

        return output_text, head_text_span, tail_text_span

    def create_representation_with_types(self, text, head_qid, head_text, tail_qid, tail_text, property):
        head_types = self.types_dictionary.get(head_qid, [])
        tail_types = self.types_dictionary.get(tail_qid, [])
        head_type_descriptions = " and ".join(
            [self.type_labels[x] for x in head_types if x in self.type_labels])
        tail_type_descriptions = " and ".join(
            [self.type_labels[x] for x in tail_types if x in self.type_labels])
        if not head_type_descriptions:
            head_type_descriptions = NO_TOKEN
        if not tail_type_descriptions:
            tail_type_descriptions = NO_TOKEN
        property_description = self.rel_to_description[property[1]]

        if self.include_types:
            output_text = f"[CLS] Given the Head Entity : {head_text} with Types : {head_type_descriptions}, Tail Entity : {tail_text} with Types : {tail_type_descriptions} and Context : {text}, the context expresses the relation [SEP] {property[0]}"
        else:
            output_text = f"[CLS] Given the Head Entity : {head_text}, Tail Entity : {tail_text} and Context : {text}, the context expresses the relation [SEP] {property[0]}"
        if self.include_descriptions:
            output_text += f" defined as {property_description} [SEP]"
        else:
            output_text += f" [SEP]"


        head_text_start = output_text.find(f"Head Entity : {head_text}") + len(f"Head Entity : ")

        head_text_span = (head_text_start, head_text_start + len(head_text))
        tail_text_start = output_text.find(f"Tail Entity : {tail_text}") + len(f"Tail Entity : ")
        tail_text_span = (tail_text_start, tail_text_start + len(tail_text))
        assert output_text[head_text_span[0]:head_text_span[1]] == head_text
        assert output_text[tail_text_span[0]:tail_text_span[1]] == tail_text

        return output_text, head_text_span, tail_text_span

    def create_representations(self, head_text, tail_text, text, elem, relation):
        head_id = elem["head_id"]
        tail_id = elem["tail_id"]
        if self.use_predicted_candidates and "head_prediction" in elem and "tail_prediction" in elem:
            head_id = elem["head_prediction"]
            tail_id = elem["tail_prediction"]
        text_representation = self.create_representation_with_types(text, head_id, head_text, tail_id,
                                                                    tail_text, relation)
        #type_tuple = self.create_type_representation(elem["head_id"], elem["tail_id"])
        type_tuple = None
        return text_representation, type_tuple

    def convert_dataset(self, dataset, all_relations, training=False, include_hard=True):
        label_dict = {x[1]:x[0] for x in all_relations}
        final_dataset = []
        for idx, item in enumerate(dataset):
            for idx_, elem in enumerate(item["triplets"]):
                text = " ".join(elem["tokens"])
                if elem["head"] and elem["tail"]:
                    head_text = " ".join(elem["tokens"][elem["head"][0]:elem["head"][-1] + 1])
                    tail_text = " ".join(elem["tokens"][elem["tail"][0]:elem["tail"][-1] + 1])
                    property = (elem["label"], elem["label_id"])
                    allowed_properties = [x for x in all_relations if x != property]
                    if training:
                        random_other_properties = []
                        if include_hard:
                            hard_relations = [x for x in self.similarity_to_other_properties[property[1]][:20]]
                            distribution = [x[1] + 0.0001 for x in hard_relations]
                            normalizer = sum(distribution)
                            distribution = [x / normalizer for x in distribution]
                            sampled_indices = np.random.choice(np.arange(0, len(distribution)), size=self.hard_other_properties,p=distribution, replace=False)
                            hard_relations = [(label_dict[relation], relation) for relation, _ in hard_relations]
                            random_other_properties += [hard_relations[x] for x in sampled_indices]
                            allowed_properties = [x for x in allowed_properties if x not in random_other_properties]
                        random_other_properties += random.sample(allowed_properties, self.other_properties)
                        all_representations = [self.create_representations(head_text, tail_text, text, elem, property)] + \
                                              [self.create_representations(head_text, tail_text, text, elem, relation) for relation in random_other_properties]
                        labels = [1] + [0] * len(random_other_properties)
                        relations = [property] + random_other_properties
                    else:
                        all_representations = []
                        labels = []
                        relations = []
                        for relation in all_relations:
                            all_representations.append(self.create_representations(head_text, tail_text, text, elem, relation))
                            if relation == property:
                                labels.append(1)
                            else:
                                labels.append(0)
                        relations += all_relations

                    final_dataset.append((all_representations, labels, relations, (idx, idx_)))
        return CustomDataset(final_dataset)

    def structure_dataset_split(self, dataset: list):
        all_relations = set()
        for item in dataset:
            for triplet in item["triplets"]:
                all_relations.add((triplet["label"], triplet["label_id"]))
        all_relations = sorted(list(all_relations))
        return dataset, all_relations
    def load_dataset(self, folder_name: str):
        train = []
        val = []
        test = []
        for item in jsonlines.open(f"{folder_name}/train_mapped.jsonl"):
            train.append(item)
        for item in jsonlines.open(f"{folder_name}/dev_mapped.jsonl"):
            val.append(item)
        test_dataset_name = "test_mapped.jsonl"
        if self.use_predicted_candidates:
            test_dataset_name = "test_mapped_predicted.jsonl"
        for item in jsonlines.open(f"{folder_name}/{test_dataset_name}"):
            test.append(item)
        return self.structure_dataset_split(train), self.structure_dataset_split(val), self.structure_dataset_split(test)

    def setup(self, stage=None):
        (train, train_relations), (val, val_relations), (test, test_relations) = self.load_dataset(self.dataset_name)
        matrix, matrix_properties = create_matrix(f"{self.dataset_name}/train_mapped.jsonl", self.types_dictionary)
        similarity_to_other_properties = {}
        for idx, property in enumerate(matrix_properties):
            elements = list(enumerate(matrix[idx, :].tolist()))
            sorted_elements = sorted(elements, key=lambda x: x[1], reverse=True)
            sorted_elements = [(matrix_properties[x[0]], x[1]) for x in sorted_elements if x[0] != idx]
            similarity_to_other_properties[property] = sorted_elements
        self.similarity_to_other_properties = similarity_to_other_properties
        self._train_dataset = train
        self.train_relations = train_relations
        self._val_dataset = val
        self.val_relations = val_relations
        self._test_dataset = test
        self.test_relations = test_relations
        self.train_dataset = self.convert_dataset(train, train_relations, training=True)
        self.val_dataset = self.convert_dataset(val, val_relations)
        self.test_dataset = self.convert_dataset(test, test_relations)
    def train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.alt_tokenizer),
                          num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args["batch_size"], shuffle=False,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.alt_tokenizer),
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args["batch_size"], shuffle=False,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.alt_tokenizer),
                          num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = RelationExtractionDataset("wiki/unseen_5_seed_0", None, {"batch_size": 2}     )
    dataset.setup()

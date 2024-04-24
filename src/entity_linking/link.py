import jsonlines
from refined.data_types.base_types import Span, Entity
from refined.inference.processor import Refined
import argparse

from tqdm import tqdm


def load_dataset(dataset_name):
    dataset = []
    counter = 0
    examples_to_skip = set()
    for item in jsonlines.open(dataset_name):
        for triplet in item["triplets"]:
            try:
                text = " ".join(triplet["tokens"])
                head_tokens = triplet["head"]
                tail_tokens = triplet["tail"]
                head_start = sum([len(t) for t in triplet["tokens"][:head_tokens[0]]]) + head_tokens[0]
                head_end = sum([len(t) for t in triplet["tokens"][:head_tokens[-1]]]) + len(triplet["tokens"][head_tokens[-1]]) + head_tokens[-1]
                tail_start = sum([len(t) for t in triplet["tokens"][:tail_tokens[0]]]) + tail_tokens[0]
                tail_end = sum([len(t) for t in triplet["tokens"][:tail_tokens[-1]]]) + len(triplet["tokens"][tail_tokens[-1]]) + tail_tokens[-1]
                spans = [Span(text[head_start: head_end], head_start, head_end-head_start,
                              gold_entity=Entity(triplet["head_id"],
                                                 wikipedia_entity_title=0)), Span(text[tail_start: tail_end], tail_start, tail_end-tail_start, gold_entity=Entity(triplet["tail_id"],
                                                                                                                                                            wikipedia_entity_title=1))]
            except:
                examples_to_skip.add(counter)
                continue
            dataset.append((text, spans))
            counter += 1
    return dataset, examples_to_skip

def process_batch(batch):
    return zip(*batch)


refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
def main(dataset_name: str, batch_size: int = 16):
    data_to_link, examples_to_skip = load_dataset(dataset_name)
    all_processed_texts = []
    for idx in tqdm(range(0,  len(data_to_link), batch_size,)):
        batch = data_to_link[idx:idx + batch_size]
        texts, spans = process_batch(batch)
        for spans_ in spans:
            _ = refined.preprocessor.add_candidates_to_spans(
                spans_
            )
        processed_texts = refined.process_text_batch(texts, spans)
        all_processed_texts.extend(processed_texts)
    counter = 0
    correct = 0
    all_elements = 0
    with jsonlines.open(dataset_name.replace(".jsonl", "_predicted.jsonl"), "w") as f:
        for item in jsonlines.open(dataset_name):
            for triplet in item["triplets"]:
                if counter in examples_to_skip:
                    continue
                for span in all_processed_texts[counter].spans:
                    if span.gold_entity.wikipedia_entity_title == 0:
                        triplet["head_prediction"] = span.predicted_entity.wikidata_entity_id
                    else:
                        triplet["tail_prediction"] = span.predicted_entity.wikidata_entity_id
                if triplet["head_prediction"] == triplet["head_id"]:
                    correct += 1
                if triplet["tail_prediction"] == triplet["tail_id"]:
                    correct += 1
                all_elements += 2
                counter += 1
            f.write(item)
    print(dataset_name)
    print(f"Accuracy: {correct/all_elements}")
    print(f"Skipped {len(examples_to_skip)} examples")




main("fewrel/unseen_5_seed_0/test_mapped.jsonl")
main("fewrel/unseen_5_seed_1/test_mapped.jsonl")
main("fewrel/unseen_5_seed_2/test_mapped.jsonl")
main("fewrel/unseen_5_seed_3/test_mapped.jsonl")
main("fewrel/unseen_5_seed_4/test_mapped.jsonl")

main("fewrel/unseen_10_seed_0/test_mapped.jsonl")
main("fewrel/unseen_10_seed_1/test_mapped.jsonl")
main("fewrel/unseen_10_seed_2/test_mapped.jsonl")
main("fewrel/unseen_10_seed_3/test_mapped.jsonl")
main("fewrel/unseen_10_seed_4/test_mapped.jsonl")

main("fewrel/unseen_15_seed_0/test_mapped.jsonl")
main("fewrel/unseen_15_seed_1/test_mapped.jsonl")
main("fewrel/unseen_15_seed_2/test_mapped.jsonl")
main("fewrel/unseen_15_seed_3/test_mapped.jsonl")
main("fewrel/unseen_15_seed_4/test_mapped.jsonl")

main("wiki/unseen_5_seed_0/test_mapped.jsonl")
main("wiki/unseen_5_seed_1/test_mapped.jsonl")
main("wiki/unseen_5_seed_2/test_mapped.jsonl")
main("wiki/unseen_5_seed_3/test_mapped.jsonl")
main("wiki/unseen_5_seed_4/test_mapped.jsonl")

main("wiki/unseen_10_seed_0/test_mapped.jsonl")
main("wiki/unseen_10_seed_1/test_mapped.jsonl")
main("wiki/unseen_10_seed_2/test_mapped.jsonl")
main("wiki/unseen_10_seed_3/test_mapped.jsonl")
main("wiki/unseen_10_seed_4/test_mapped.jsonl")

main("wiki/unseen_15_seed_0/test_mapped.jsonl")
main("wiki/unseen_15_seed_1/test_mapped.jsonl")
main("wiki/unseen_15_seed_2/test_mapped.jsonl")
main("wiki/unseen_15_seed_3/test_mapped.jsonl")
main("wiki/unseen_15_seed_4/test_mapped.jsonl")



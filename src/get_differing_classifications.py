import argparse
import json

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.dataset import RelationExtractionDataset
from src.relation_extractor import RelationExtractor


def compare(predictions_1, predictions_2, test_dataset, filename: str):
    predictions_1, gold_labels, batch_relations = zip(*predictions_1)
    predictions_2, _, _ = zip(*predictions_2)
    main_relations = batch_relations[0][0]
    predictions_1 = np.concatenate(predictions_1)
    predictions_2 = np.concatenate(predictions_2)
    gold_labels = np.concatenate(gold_labels)
    all_examples = []
    for item in test_dataset.dataset:
        all_examples.append(item[0][0][0])
    prediction_1_correct_but_not_2 = []
    prediction_2_correct_but_not_1 = []
    both_incorrect = []
    for prediction_1, prediction_2, gold_label, example in zip(predictions_1, predictions_2, gold_labels, all_examples):
        if prediction_1 == gold_label and prediction_2 != gold_label:
            prediction_1_correct_but_not_2.append([example, main_relations[prediction_2], main_relations[gold_label]])
        elif prediction_1 != gold_label and prediction_2 == gold_label:
            prediction_2_correct_but_not_1.append([example, main_relations[prediction_1], main_relations[gold_label]])
        elif prediction_1 != gold_label and prediction_2 != gold_label:
            both_incorrect.append([example, main_relations[gold_label], main_relations[prediction_1], main_relations[prediction_2]])
    accuracy_1 = sum(predictions_1 == gold_labels) / len(gold_labels)
    accuracy_2 = sum(predictions_2 == gold_labels) / len(gold_labels)
    print(f"Accuracy 1: {accuracy_1}")
    print(f"Accuracy 2: {accuracy_2}")
    json.dump({
        "prediction_1_correct_but_not_2": prediction_1_correct_but_not_2,
        "prediction_2_correct_but_not_1": prediction_2_correct_but_not_1,
        "both_incorrect": both_incorrect
    }, open(f"compare_{accuracy_1}_{accuracy_2}_{filename}.json", "w"), indent=4)
def main(args):
    rel_extractor = RelationExtractor.load_from_checkpoint(args.model_checkpoint,model_type=args.model_type,
                                              add_additional_layer=args.add_additional_layer, strict=False)
    rel_extractor_2 = RelationExtractor.load_from_checkpoint(args.model_checkpoint_2,model_type=args.model_type,
                                                add_additional_layer=args.add_additional_layer, strict=False)
    dataset = RelationExtractionDataset(args.dataset_name, rel_extractor.tokenizer,
                                        args={"batch_size": args.batch_size,
                                         "num_workers": args.num_workers,
                                         "replace_types": args.replace_types,
                                         "other_properties": args.other_properties,
                                         "hard_other_properties": args.hard_other_properties,
                                         "use_alternative_types": args.use_alternative_types,
                                                                                      "include_descriptions": args.include_descriptions,
                                                                                    "include_types": args.include_types,
                                              "use_predicted_candidates": args.use_predicted_candidates})
    dataset_2 = RelationExtractionDataset(args.dataset_name, rel_extractor.tokenizer,
                                        args={"batch_size": args.batch_size,
                                              "num_workers": args.num_workers,
                                              "replace_types": args.replace_types,
                                              "other_properties": args.other_properties,
                                              "hard_other_properties": args.hard_other_properties,
                                              "use_alternative_types": args.use_alternative_types,
                                              "use_predicted_candidates": args.use_predicted_candidates})
    wandb_logger = WandbLogger(project="zs_relation_extractor", name=f"evaluate_{args.dataset_name}")

    args_dict = vars(args)
    wandb_logger.log_hyperparams(args_dict)

    trainer = Trainer(logger=wandb_logger,
                      accumulate_grad_batches=args.accumulate_grad_batches)
    dataset.setup()
    dataset_2.setup()
    predictions_1 = trainer.predict(rel_extractor, dataloaders=dataset.test_dataloader())
    predictions_2 = trainer.predict(rel_extractor_2, dataloaders=dataset_2.test_dataloader())
    filename = args.dataset_name.split("/")[-1]
    compare(predictions_1, predictions_2, dataset_2.test_dataset, filename)









if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_checkpoint", type=str, required=True)
    argparser.add_argument("--model_checkpoint_2", type=str, required=True)
    argparser.add_argument("--dataset_name", type=str, default="wiki/unseen_5_seed_0")
    argparser.add_argument("--model_type", type=str, default="distilbert-base-uncased")
    argparser.add_argument("--batch_size", type=int, default=24)
    argparser.add_argument("--num_workers", type=int, default=2)
    argparser.add_argument("--accumulate_grad_batches", type=int, default=1)
    argparser.add_argument("--replace_types", action="store_true", default=False)
    argparser.add_argument("--other_properties", type=int, default=5)
    argparser.add_argument("--add_additional_layer", action="store_true")
    argparser.add_argument("--hard_other_properties", type=int, default=5)
    argparser.add_argument("--include_descriptions", action="store_true", default=False)
    argparser.add_argument("--include_types", action="store_true", default=False)
    argparser.add_argument("--use_alternative_types", action="store_true", default=False)
    argparser.add_argument("--use_predicted_candidates", action="store_true", default=False)


    args = argparser.parse_args()
    main(args)
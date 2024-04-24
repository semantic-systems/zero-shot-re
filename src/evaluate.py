import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.dataset import RelationExtractionDataset
from src.relation_extractor import RelationExtractor


def main(args):
    rel_extractor = RelationExtractor.load_from_checkpoint(args.model_checkpoint,model_type=args.model_type, strict=False)
    dataset = RelationExtractionDataset(args.dataset_name, rel_extractor.tokenizer,
                                        args={"batch_size": args.batch_size,
                                         "num_workers": args.num_workers,
                                         "other_properties": args.other_properties,
                                         "hard_other_properties": args.hard_other_properties,
                                                                                      "include_descriptions": args.include_descriptions,
                                                                                    "include_types": args.include_types,
                                              "use_predicted_candidates": args.use_predicted_candidates})
    wandb_logger = WandbLogger(project="zs_relation_extractor", name=f"evaluate_{args.dataset_name}")

    args_dict = vars(args)
    wandb_logger.log_hyperparams(args_dict)

    trainer = Trainer(logger=wandb_logger,
                      accumulate_grad_batches=args.accumulate_grad_batches)

    trainer.test(rel_extractor, datamodule=dataset)







if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_checkpoint", type=str, required=True)
    argparser.add_argument("--dataset_name", type=str, default="wiki/unseen_5_seed_0")
    argparser.add_argument("--model_type", type=str, default="distilbert-base-uncased")
    argparser.add_argument("--batch_size", type=int, default=24)
    argparser.add_argument("--num_workers", type=int, default=2)
    argparser.add_argument("--accumulate_grad_batches", type=int, default=1)
    argparser.add_argument("--other_properties", type=int, default=5)
    argparser.add_argument("--hard_other_properties", type=int, default=0)
    argparser.add_argument("--include_descriptions", action="store_true", default=False)
    argparser.add_argument("--include_types", action="store_true", default=False)
    argparser.add_argument("--use_predicted_candidates", action="store_true", default=False)


    args = argparser.parse_args()
    main(args)
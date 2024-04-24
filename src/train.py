import argparse
import os
import random

import numpy as np
import torch
import datetime
import wandb
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from dataset import RelationExtractionDataset
from relation_extractor import RelationExtractor
from pytorch_lightning import Trainer


def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(directory, dataset_name, seed, args, checkpoint=None):
    seed_dataset_name = f"{dataset_name}_seed_{seed}"
    rel_extractor = RelationExtractor(model_type=args.model_type,
                                      learning_rate=args.lr)
    alt_tokenizer = None
    if checkpoint is not None:
        alt_tokenizer = rel_extractor.type_model.tokenizer
    dataset = RelationExtractionDataset(seed_dataset_name, rel_extractor.tokenizer, alt_tokenizer, {"batch_size": args.batch_size,
                                                                                     "num_workers": args.num_workers,
                                                                                     "replace_types": args.replace_types,
                                                                                     "other_properties": args.other_properties,
                                                                                     "hard_other_properties": args.hard_other_properties,
                                                                                     "use_alternative_types": args.use_alternative_types,
                                                                                      "include_descriptions": args.include_descriptions,
                                                                                    "include_types": args.include_types})

    wandb_logger = WandbLogger(project="zs_relation_extractor", name=seed_dataset_name)
    args_dict = vars(args)
    args_dict["seed"] = seed
    args_dict["dataset"] = seed_dataset_name
    wandb_logger.log_hyperparams(args_dict)
    checkpoint_callback_2 = ModelCheckpoint(
        monitor='val_f1',
        dirpath=f"{directory}/{seed_dataset_name}",
        filename='model-{epoch:02d}-{val_f1:.2f}_val_f1',
        save_top_k=1,
        mode='max',
    )
    early_stopping_callback = EarlyStopping(monitor="val_f1", mode="max")

    strategy = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto"
    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback_2,
                                                early_stopping_callback], logger=wandb_logger,
                      accumulate_grad_batches=args.accumulate_grad_batches, strategy=strategy,
                      deterministic=True
                      )
    trainer.fit(rel_extractor, dataset)
    wandb_logger.log_metrics({"best_val_f1": checkpoint_callback_2.best_model_score})
def main(args):
    torch.set_float32_matmul_precision('high') # 'medium'
    seed_everything(42, workers=True)
    seeds = args.seeds
    dataset_name = args.dataset_name
    directory = f"relation_extractor_{dataset_name}"
    if os.path.exists(directory):
        directory = f"relation_extractor_{dataset_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    for seed in seeds:
        run_experiment(directory, dataset_name, seed, args)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, default="fewrel/unseen_5")
    argparser.add_argument("--model_type", type=str, default="bert-base-cased")
    argparser.add_argument("--batch_size", type=int, default=24)
    argparser.add_argument("--num_workers", type=int, default=2)
    argparser.add_argument("--accumulate_grad_batches", type=int, default=2)
    argparser.add_argument("--replace_types", action="store_true", default=False)
    argparser.add_argument("--lr", type=float, default= 5e-5)
    argparser.add_argument("--alternative_lr", type=float, default=None)
    argparser.add_argument("--num_layers", type=int, default=1)
    argparser.add_argument("--seeds", type=int, default=[0,1,2,3,4], nargs="+")
    argparser.add_argument("--other_properties", type=int, default=5)
    argparser.add_argument("--hard_other_properties", type=int, default=0)
    argparser.add_argument("--include_descriptions", action="store_true", default=False)
    argparser.add_argument("--include_types", action="store_true", default=False)
    argparser.add_argument("--use_alternative_types", action="store_true", default=False)


    args = argparser.parse_args()
    main(args)
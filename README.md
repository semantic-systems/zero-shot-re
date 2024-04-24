#  Code of ESWC 2024 submission "Incorporating Type Information Into Zero-Shot Relation Extraction"


## Steps to reproduce

1. Unpack the following files:
    * [wiki.tar.gz](wiki.tar.gz)
    * [fewrel.tar.gz](fewrel.tar.gz)
    * [entity_types_reparsed.pickle.gz](entity_types_reparsed.pickle.gz)
    * [types_to_description_alt.json.gz](types_to_description_alt.json.gz)
    * [rel_id_title_description_cleaned.jsonl.gz](rel_id_title_description_cleaned.jsonl.gz)
    * [types_to_label.json.gz](types_to_label.json.gz)
2. Run the following command to train the model for each dataset:
    ```
    python3 src/train.py ...
    ```
   
3. Evaluate on each seed of each dataset by using the following command:
    ```
    python3 src/evaluate.py ...
    ```
   

## Functions
Arguments for `train.py`:

| Argument                              | Type          | Default Value     | Description                                                                                                        |
|---------------------------------------|---------------|-------------------|--------------------------------------------------------------------------------------------------------------------|
| --dataset_name                        | str           | "fewrel/unseen_5" | Specifies the name of the dataset. This executes the training for all seeds as specified by the --seeds parameter. |
| --model_type                          | str           | "bert-base-cased" | Specifies the type of model to be used.                                                                            |
| --batch_size                          | int           | 24                | Sets the batch size for training.                                                                                  |
| --num_workers                         | int           | 2                 | Number of worker processes for data loading.                                                                       |
| --accumulate_grad_batches             | int           | 2                 | Accumulates gradients over a specified number of batches.                                                          |
| --lr                                  | float         | 5e-5              | Learning rate for optimization.                                                                                    |
| --seeds                               | int, List     | [0, 1, 2, 3, 4]   | List of seeds of the dataset to train on.                                                                          |
| --include_descriptions                | store_true    | False             | Includes descriptions in the textual representation if this flag is present.                                       |
| --include_types                       | store_true    | False             | Includes types in the textual if this flag is present.                                                             |


Arguments for `evaluate.py`:

| Argument                              | Type          | Default Value                  | Required | Description                                                    |
|---------------------------------------|---------------|--------------------------------|----------|----------------------------------------------------------------|
| --model_checkpoint                    | str           | -                              | Yes      | Specifies the path to the model checkpoint.                    |
| --dataset_name                        | str           | "fewrel/unseen_5_seed_0"         | No       | Specifies the name of the dataset with the corresponding seed. |
| --model_type                          | str           | "bert-base-cased"      | No       | Specifies the type of model to be used.                        |
| --batch_size                          | int           | 24                             | No       | Sets the batch size for training.                              |
| --num_workers                         | int           | 2                              | No       | Number of worker processes for data loading.                   |
| --accumulate_grad_batches             | int           | 1                              | No       | Accumulates gradients over a specified number of batches.      |
| --other_properties                    | int           | 5                              | No       | Specifies the value for some other properties.                 |
| --hard_other_properties               | int           | 0                              | No       | Specifies the value for some other hard properties.            |
| --include_descriptions                | store_true    | False                          | No       | Includes descriptions in the textual representation if this flag is present.                     |
| --include_types                       | store_true    | False                          | No       | Includes types in the textual if this flag is present.                                      |
| --use_predicted_candidates            | store_true    | False                          | No       | Uses predicted candidates if this flag is present.             |





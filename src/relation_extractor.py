import numpy
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer


class RelationExtractor(pl.LightningModule):
    def __init__(self, model_type: str, learning_rate: float=5e-5):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.learning_rate = learning_rate

        self.loss = CrossEntropyLoss()
        self.scoring = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.model.config.hidden_size, 1)
        )
        self.empty_label = None
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def average(self, outputs, indices):
        if len(indices.size()) > 1  and indices.size(1) == 2:
            indices = indices[:, 1]
        # Compute the unique indices and their counts
        unique_indices, counts = torch.unique(indices, return_counts=True)

        # Use scatter_add to compute the sum of rows for each unique index
        sums = torch.zeros(unique_indices.size(0), outputs.size(1), dtype=outputs.dtype, device=outputs.device)
        indices = indices.unsqueeze(1).expand(-1, outputs.size(1))
        sums.scatter_add_(0, indices, outputs)

        # Divide the sums by the counts to get the averages
        averaged_tensors = sums / counts.unsqueeze(1).float()
        return averaged_tensors

    def forward(self, input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions,
                       tail_token_positions, head_type_indices,
                       tail_type_indices, head_type_string_indices,
                       tail_type_string_indices,**kwargs):
        embedded = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = embedded[0][:, 0, :]
        scores = self.scoring(cls_embedding)
        if type_input_ids is None:
            return scores, None

        type_embedded = self.type_embedder(input_ids=type_input_ids, attention_mask=type_attention_mask)
        type_cls_embedding = type_embedded[0][:, 0, :]
        head_type_embeddings = type_cls_embedding[head_type_string_indices]
        tail_type_embeddings = type_cls_embedding[tail_type_string_indices]
        averaged_head_type_embeddings = self.average(head_type_embeddings, head_type_indices)
        averaged_tail_type_embeddings = self.average(tail_type_embeddings, tail_type_indices)


        alt_scores = self.scoring(torch.cat([cls_embedding, averaged_head_type_embeddings, averaged_tail_type_embeddings], dim=1))
        return scores, alt_scores


    def unpack_batch(self, batch):
        tokenized, tokenized_types, labels, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, maximum_elements, batch_relations, all_indices = batch
        if tokenized_types is not None:
            type_input_ids = tokenized_types["input_ids"].to(self.device)
            type_attention_mask = tokenized_types["attention_mask"].to(self.device)
        else:
            type_input_ids = None
            type_attention_mask = None
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        return input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, labels, maximum_elements, batch_relations, all_indices
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, labels, maximum_elements, batch_relations, all_indices = self.unpack_batch(batch)
        outputs, alt_scores = self(input_ids=input_ids, attention_mask=attention_mask,
                       type_input_ids=type_input_ids,
                       type_attention_mask=type_attention_mask,
                       head_token_positions=head_token_positions,
                       tail_token_positions=tail_token_positions, head_type_indices=head_type_indices,
                       tail_type_indices=tail_type_indices, head_type_string_indices=head_type_string_indices,
                       tail_type_string_indices=tail_type_string_indices)
        outputs = torch.reshape(outputs, (-1, maximum_elements))
        labels = torch.reshape(labels, (-1, maximum_elements))
        predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        gold_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
        return predictions, gold_labels, batch_relations
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, labels, maximum_elements, batch_relations, all_indices = self.unpack_batch(
            batch)
        outputs, alt_scores = self(input_ids=input_ids, attention_mask=attention_mask,
                       type_input_ids=type_input_ids,
                       type_attention_mask=type_attention_mask,
                       head_token_positions=head_token_positions,
                       tail_token_positions=tail_token_positions, head_type_indices=head_type_indices,
                       tail_type_indices=tail_type_indices, head_type_string_indices=head_type_string_indices,
                       tail_type_string_indices=tail_type_string_indices)
        outputs = torch.reshape(outputs, (-1, maximum_elements))

        labels = torch.reshape(labels, (-1, maximum_elements))
        loss = self.loss(outputs, labels)

        if alt_scores is not None:
            alt_scores = torch.reshape(alt_scores, (-1, maximum_elements))
            alt_loss = self.loss(alt_scores, labels)
            loss = (loss + alt_loss) / 2
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def calculate_metrics(self, tp, tp_fp, tp_fn, count):
        precision = tp / tp_fp
        recall = tp / tp_fn
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = tp / count
        return precision, recall, f1, accuracy

    @staticmethod
    def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
        '''
        This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
        code borrowed from the following link:
        https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
        '''
        if i == -1:
            i = len(predicted_idx)

        complete_rel_set = set(gold_idx) - {empty_label}
        avg_prec = 0.0
        avg_rec = 0.0

        for r in complete_rel_set:
            r_indices = (predicted_idx[:i] == r)
            tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
            tp_fp = len(r_indices.nonzero()[0])
            tp_fn = len((gold_idx == r).nonzero()[0])
            prec = (tp / tp_fp) if tp_fp > 0 else 0
            rec = tp / tp_fn
            avg_prec += prec
            avg_rec += rec
        f1 = 0
        avg_prec = avg_prec / len(set(predicted_idx[:i]))
        avg_rec = avg_rec / len(complete_rel_set)
        if (avg_rec + avg_prec) > 0:
            f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

        accuracy = len((predicted_idx[:i] == gold_idx[:i]).nonzero()[0]) / len(predicted_idx[:i])
        return avg_prec, avg_rec, f1, accuracy

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, labels, maximum_elements, batch_relations, all_indices = self.unpack_batch(
            batch)
        outputs, alt_scores = self(input_ids=input_ids, attention_mask=attention_mask,
                       type_input_ids=type_input_ids,
                       type_attention_mask=type_attention_mask,
                       head_token_positions=head_token_positions,
                       tail_token_positions=tail_token_positions, head_type_indices=head_type_indices,
                       tail_type_indices=tail_type_indices, head_type_string_indices=head_type_string_indices,
                       tail_type_string_indices=tail_type_string_indices)
        # Concatenate according to batch index
        outputs = torch.reshape(outputs, (-1, maximum_elements))
        labels = torch.reshape(labels, (-1, maximum_elements))

        loss = self.loss(outputs, labels)

        predicted = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        gold_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
        self.validation_step_outputs.append((loss, predicted, gold_labels))

        if alt_scores is not None:
            alt_scores = torch.reshape(alt_scores, (-1, maximum_elements))
            alt_loss = self.loss(alt_scores, labels)
            loss = (loss + alt_loss) / 2
        return loss, predicted, gold_labels

    def on_validation_epoch_end(self):
        loss, predicted, gold_labels =  zip(*self.validation_step_outputs)
        loss = torch.mean(torch.stack(loss))
        predicted = numpy.concatenate(predicted)
        gold_labels = numpy.concatenate(gold_labels)


        precision, recall, f1, accuracy = self.compute_macro_PRF(predicted, gold_labels)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_accuracy', accuracy, sync_dist=True)
        self.log('val_precision', precision, sync_dist=True)
        self.log('val_recall', recall, sync_dist=True)
        self.log('val_f1', f1, sync_dist=True)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, type_input_ids, type_attention_mask, head_token_positions, tail_token_positions, head_type_indices, tail_type_indices, head_type_string_indices, tail_type_string_indices, labels, maximum_elements, batch_relations, all_indices = self.unpack_batch(
            batch)
        outputs, _ = self(input_ids=input_ids, attention_mask=attention_mask,
                       type_input_ids=type_input_ids,
                       type_attention_mask=type_attention_mask,
                       head_token_positions=head_token_positions,
                       tail_token_positions=tail_token_positions, head_type_indices=head_type_indices,
                       tail_type_indices=tail_type_indices, head_type_string_indices=head_type_string_indices,
                       tail_type_string_indices=tail_type_string_indices)
        outputs = torch.reshape(outputs, (-1, maximum_elements))
        labels = torch.reshape(labels, (-1, maximum_elements))
        predicted = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        gold_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()
        self.test_step_outputs.append((predicted, gold_labels))
        return predicted, gold_labels

    def on_test_epoch_end(self):
        predicted, gold_labels = zip(*self.test_step_outputs)
        predicted = numpy.concatenate(predicted)
        gold_labels = numpy.concatenate(gold_labels)

        precision, recall, f1, accuracy = self.compute_macro_PRF(predicted, gold_labels)
        self.log('test_accuracy', accuracy, sync_dist=True)
        self.log('test_precision', precision, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_f1', f1, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

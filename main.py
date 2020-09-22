import os
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from vncorenlp import VnCoreNLP

from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

"""
    Model shortcut:
        - bert-base-multilingual-uncased
        - bert-base-multilingual-cased
        - xlm-roberta-base
        - xlm-mlm-xnli15-1024
        - xlm-mlm-tlm-xnli15-1024
        - roberta-large-mnli
        - facebook/bart-large-mnli
        - vinai/phobert-base
        - vinai/phobert-large
        
    Data shortcut:
        - VLSP2016
        - AIVIVN
        - UIT-VSFC
"""

PHOBERT_ALIASES = [
    'vinai/phobert-base',
    'vinai/phobert-large'
]

TRANSFORMER_ALIASES = [
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
    'xlm-mlm-xnli15-1024',
    'xlm-mlm-tlm-xnli15-1024',
    'roberta-large-mnli',
    'facebook/bart-large-mnli',
]

rdrsegmenter = None


def rdr_word_segmenter(text):
    sentences = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(sent) for sent in sentences])
    return text


def load_data(root='data', data='VLSP2016', is_train=True, is_phobert=False):
    if data == 'VLSP2016':
        df = pd.read_csv(os.path.join(root, data, 'SA-2016.train' if is_train else 'SA-2016.test'),
                         names=['sentence', 'label'],
                         sep='\t',
                         encoding='utf-8-sig')

        texts, labels = df['sentence'].apply(lambda row: row.strip()).tolist(), \
                        df['label'].apply(lambda row: 0 if row == 'NEG' else 1 if row == 'NEU' else 2).tolist()
    elif data == 'UIT-VSFC':
        sentences = open(os.path.join(root, data, 'train' if is_train else 'test', 'sents.txt'),
                         mode='r',
                         encoding='utf-8-sig').read().strip().split('\n')
        sentiment = open(os.path.join(root, data, 'train' if is_train else 'test', 'sentiments.txt'),
                         mode='r',
                         encoding='utf-8-sig').read().strip().split('\n')
        texts, labels = [text.strip() for text in sentences], [int(label) for label in sentiment]
    else:
        pivot = 0.9
        file_reader = open(os.path.join(root, data, 'train.crash'), mode='r', encoding='utf-8-sig').read().strip()
        file_reader = file_reader.split('\n\ntrain_')
        texts, labels = [sent[8: -3].strip() for sent in file_reader], [int(sent[-1:]) for sent in file_reader]
        texts, labels = texts[: int(len(texts) * pivot)] if is_train else texts[int(len(texts) * pivot):], \
                        labels[: int(len(labels) * pivot)] if is_train else labels[int(len(labels) * pivot):]

    if is_phobert is True:
        texts = [rdr_word_segmenter(text) for text in tqdm(texts, 'Using RDR Segmenter for PhoBERT')]

    return texts, labels


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

        assert hasattr(self.encodings, 'input_ids'), TypeError('There is no input_ids in sentence encoding.')
        assert hasattr(self.encodings, 'attention_mask'), TypeError('There is no attention_mask in sentence encoding.')
        assert self.encodings['input_ids'].__len__() == self.labels.__len__(), IndexError

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.labels.__len__()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    # argument parsing
    argument_parser = argparse.ArgumentParser(description='Fine-tune transformer models for '
                                                          'Vietnamese Sentiment Analysis.')
    argument_parser.add_argument('--model', type=str, default='vinai/phobert-base',
                                 help='Model shortcut from pretrained hub.')
    argument_parser.add_argument('--freeze_encoder', type=bool, default=True,
                                 help='Whether BERT base encoder is freeze or not.')
    argument_parser.add_argument('--epoch', type=int, default=1, help='Number of training epochs.')
    argument_parser.add_argument('--learning_rate', type=int, default=1e-5, help='Model learning rate.')
    argument_parser.add_argument('--accumulation_steps', type=int, default=50, help='Gradient accumulation steps.')
    argument_parser.add_argument('--device', type=str, default='cuda', help='Training device.')
    argument_parser.add_argument('--root', type=str, default='data', help='Directory to dataset.')
    argument_parser.add_argument('--data', type=str, default='VLSP2016',
                                 help='Which dataset use to train a.k.a (VLSP2016, UIT-VSFC, AIVIVN).')
    argument_parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
    argument_parser.add_argument('--max_length', type=int, default=256, help='Maximum length of BERT tokenizer.')
    argument_parser.add_argument('--num_labels', type=int, default=3,
                                 help='Number of classification labels (a.k.a sentiment polarities)')
    argument_parser.add_argument('--warmup_steps', type=int, default=300, help='Learning rate warming up step.')
    argument_parser.add_argument('--weight_decay', type=float, default=0.01, help='Training weight decay.')
    argument_parser.add_argument('--save_steps', type=int, default=10, help='Number of step to save model.')
    argument_parser.add_argument('--eval_steps', type=int, default=100, help='Number of step to evaluate model.')
    argument_parser.add_argument('--logging_steps', type=int, default=10, help='Number of step to write log.')
    args = argument_parser.parse_args()
    print(args)

    if args.model in PHOBERT_ALIASES:
        rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg",
                                 max_heap_size='-Xmx500m')

    # load sentiment data
    train_texts, train_labels = load_data(root=args.root, data=args.data, is_train=True,
                                          is_phobert=args.model in PHOBERT_ALIASES)
    test_texts, test_labels = load_data(root=args.root, data=args.data, is_train=False,
                                        is_phobert=args.model in PHOBERT_ALIASES)
    assert train_texts.__len__() == train_labels.__len__(), IndexError
    assert test_texts.__len__() == test_labels.__len__(), IndexError

    # init model
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = args.num_labels

    net = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # encode data
    train_encodings, test_encodings = tokenizer(train_texts,
                                                truncation=True,
                                                padding=True,
                                                max_length=args.max_length), \
                                      tokenizer(test_texts,
                                                truncation=True,
                                                padding=True,
                                                max_length=args.max_length)
    train_dataset, test_dataset = SentimentAnalysisDataset(train_encodings, train_labels), \
                                  SentimentAnalysisDataset(test_encodings, test_labels)

    # freeze encoder
    if args.freeze_encoder is True:
        for param in net.base_model.parameters():
            param.requires_grad = True
    print(f'Loaded Model Architecture: {args.model}')

    # init optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluate_during_training=True,
        eval_steps=args.eval_steps,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.accumulation_steps,
        # save_steps=args.save_steps,
        # no_cuda=False if args.device == 'cuda' else True
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    train_output = trainer.train()
    evaluate_output = trainer.evaluate(eval_dataset=test_dataset)

    print(train_output)
    print(evaluate_output)

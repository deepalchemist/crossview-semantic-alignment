# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-04-22 07:51:43
# Last Change:  2024-04-24 03:34:45

from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers import AutoTokenizer, BertTokenizer

model_name = 'pretrained/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, local_files_only=True)
vocab_path = 'pretrained/bert-base-uncased/vocab.txt'
bert_tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True, local_files_only=True)

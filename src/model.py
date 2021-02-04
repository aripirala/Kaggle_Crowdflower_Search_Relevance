import torch.nn as nn
import transformers
import configs

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = configs.BERT_PATH
        self.bert = configs.BERT_MODEL
        self.bert_drop = nn.Dropout(0.25)
        self.out = nn.Linear(768, 4)

    def forward(self, ids, attention_mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        o2 = self.bert_drop(o2)
        logits = self.out(o2)
        return logits

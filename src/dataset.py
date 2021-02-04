import configs
import torch

class CrowdFlowerDataset:
    def __init__(self, query, prod_title, prod_description, targets):
        self.query = query
        self.prod_title = prod_title
        self.prod_desc = prod_description
        self.targets = targets
        self.tokenizer = configs.TOKENIZER
        self.max_len = configs.MAX_LEN
        self.len = len(query)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        a_query = str(self.query[idx])
        a_prod_title = str(self.prod_title[idx])
        a_prod_desc = str(self.prod_desc[idx])
        a_target = self.targets[idx]

        inputs = self.tokenizer.encode_plus(
            a_query,
            a_prod_title + " " + a_prod_desc,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attn_mask = inputs['attention_mask']

        padding_len = self.max_len - len(ids)
        padding = [0]*padding_len
        ids = ids + padding
        token_type_ids  = token_type_ids + padding
        attn_mask = attn_mask + padding

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids':torch.tensor(token_type_ids, dtype=torch.long),
            'attn_mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(a_target, dtype=torch.long)
        }

if __name__ == '__main__':
    import pandas as pd

    data_df =  pd.read_csv('../input/train.csv')
    data_df = data_df.fillna('-999999')

    X_cols = ['query', 'product_title', 'product_description']
    train_X = data_df[X_cols]
    train_y = data_df.median_relevance.values

    train_dataset = CrowdFlowerDataset(
        query=data_df['query'].values,
        prod_title=data_df['product_title'].values,
        prod_description=data_df['product_description'].values,
        targets=train_y
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.TRAIN_BATCH_SIZE,
        shuffle=True
    )

    for batch_idx, data in enumerate(train_loader):
        print(data)
        break
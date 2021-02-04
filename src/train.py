from metrics import quadratic_weighted_kappa, loss_fn
import torch
import configs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import CrowdFlowerDataset
from model import BERTBaseUncased

import time


def train_loop_fn(data_loader, model, optimizer, scheduler=None):
    model.train()
    device = configs.DEVICE
    losses = []
    for batch_idx, data in enumerate(data_loader):
        ids = data['ids']
        mask = data['attn_mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']
        targets = targets.long()
        # print(targets)

        if device:
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print(f'preds shape = {outputs.shape}')
        # print(f'targets shape = {targets.shape}')


        # print(outputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        losses.append(batch_loss)
        if scheduler:
            scheduler.step()
    return np.mean(losses)

def eval_loop_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_preds = []
    losses = []
    device = configs.DEVICE
    for batch_idx, data in enumerate(data_loader):
        ids = data['ids']
        mask = data['attn_mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']

        if device:
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        fin_targets.append(targets.cpu().detach().numpy())
        fin_preds.append(preds.cpu().detach().numpy())

    return np.vstack(fin_preds), np.vstack(fin_targets), np.mean(losses)


def run():
    data_df = pd.read_csv('../input/train.csv')
    train_df, valid_df = train_test_split(data_df, random_state=42, test_size=0.1)
    train_df = train_df.reset_index(drop=True).sample(frac=0.1)
    valid_df = valid_df.reset_index(drop=True).sample(frac=0.1)

    train_y = train_df['median_relevance'].values
    valid_y = valid_df['median_relevance'].values

    train_dataset = CrowdFlowerDataset(
        query=train_df['query'].values,
        prod_title=train_df['product_title'].values,
        prod_description=train_df['product_description'].values,
        targets=train_y
    )
    valid_dataset = CrowdFlowerDataset(
        query=valid_df['query'].values,
        prod_title=valid_df['product_title'].values,
        prod_description=valid_df['product_description'].values,
        targets=valid_y
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= configs.TRAIN_BATCH_SIZE,
        shuffle=True
    )


    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size= configs.VALID_BATCH_SIZE,
        shuffle=False
    )

    num_train_steps = int(len(train_dataset)/ configs.TRAIN_BATCH_SIZE * configs.EPOCHS)
    device= configs.DEVICE
    model = BERTBaseUncased().to(device)
    optimizer = configs.OPTIMIZER(model.parameters(), lr=configs.LR)
    scheduler = configs.SCHEDULER(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    for epoch in range(configs.EPOCHS):

        epoch_start = time.time()

        epoch_train_loss = train_loop_fn(train_dataloader, model, optimizer, scheduler)
        outputs, targets, epoch_valid_loss = eval_loop_fn(valid_dataloader, model)

        epoch_end = time.time()
        epoch_time_elapsed =  (epoch_end - epoch_start)/60.0
        print(f'time take to run a epoch - {epoch_time_elapsed}')
        print(f'Epoch - Training loss - {epoch_train_loss} Valid loss - {epoch_valid_loss}')

        qw_kappa = quadratic_weighted_kappa(targets.squeeze().numpy(), outputs.numpy())
        print(f'Quadratic Weighted Kappa: {qw_kappa}')

if __name__ == '__main__':
    run()


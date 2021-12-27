import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import config

def train_fn(data_loader, model, optimizer, device, scheduler = None):
    model.train()
    final_loss = 0
    # for data in tqdm(data_loader, total=len(data_loader)):
    # for k, v in data.items():
    #         data[k] = v.to(device)
    # logits = model(**data)
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        
        # loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, config.NUM_LABELs), label_ids.view(-1))
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        final_loss += loss.item()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            # BertAdam hadles this automatically, use this when using AdamW optimizer
            # scheduler.step() 
        
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    # for data in tqdm(data_loader, total=len(data_loader)):
    #     for k, v in data.items():
    #         data[k] = v.to(device)
    #     _, _, loss = model(**data)
    #     final_loss += loss.item()

    preds = []
    for step, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, config.NUM_LABELs), label_ids.view(-1))
        final_loss += loss.mean().item()

        preds.extend(logits.detach().cpu().numpy())

    return final_loss / len(data_loader), preds
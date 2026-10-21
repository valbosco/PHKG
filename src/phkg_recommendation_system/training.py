import copy
import torch

def train_one_epoch(model, train_df, optimizer, criterion,
                    forward_fn, target_fn):
    model.train()
    total_loss = 0.0
    ordered = train_df.sort_values("target_date")
    for _, row in ordered.iterrows():
        optimizer.zero_grad()
        target = target_fn(row["target"]).to(next(model.parameters()).device)
        logits = forward_fn(model, row)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ordered)

def evaluate_loss(model, data_df, criterion, forward_fn, target_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _, row in data_df.sort_values("target_date").iterrows():
            target = target_fn(row["target"]).to(next(model.parameters()).device)
            logits = forward_fn(model, row)
            total_loss += criterion(logits, target).item()
    return total_loss / len(data_df)

def select_by_validation_ndcg5(model, train_df, val_df, optimizer,
                                criterion, forward_fn, target_fn,
                                prediction_fn, metric_fn,
                                max_epochs=100, patience=10):
    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_df, optimizer, criterion, forward_fn, target_fn
        )
        val_loss = evaluate_loss(
            model, val_df, criterion, forward_fn, target_fn
        )
        val_predictions = prediction_fn(model, val_df)
        val_ndcg5 = metric_fn(val_predictions, 5)["NDCG"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ndcg5": val_ndcg5
        })

        if val_ndcg5 > best_score:
            best_score = val_ndcg5
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch, best_score

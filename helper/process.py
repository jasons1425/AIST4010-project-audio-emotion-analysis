import torch
import time
import copy
import numpy as np


def train_model(model, train_loader, criterion, optimizer, epochs,
                device="cpu", valid_loader=None, scheduler=None, half=False):
    since = time.time()
    losses, loaders = {"train": []}, {"train": train_loader}
    if valid_loader:
        losses["valid"] = []
        loaders["valid"] = valid_loader
        best_model_wts = copy.deepcopy(model.state_dict())
        least_loss = np.inf

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")
        print("="*10)
        for phase in loaders:
            loader = loaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss, running_acc = 0.0, 0.0
            print(f"{phase}", end=" - ")

            with torch.set_grad_enabled(phase == "train"):
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    if half:
                        inputs, labels = inputs.half(), labels.half()
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                epoch_loss = running_loss / len(loader.dataset)
                losses[phase].append(epoch_loss)
                print(f"loss: {epoch_loss:.5f}")

                if phase == "valid":
                    if least_loss > epoch_loss:
                        least_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()

    time_elapsed = time.time() - since
    print(f"training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    if 'valid' in loaders:
        print(f"Least loss: {least_loss:.4f}")
        model.load_state_dict(best_model_wts)

    return model, losses


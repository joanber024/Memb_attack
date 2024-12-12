import time
import sys

from copy import deepcopy
import torch
import torch.nn.functional as F




def train_model(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        num_epochs,
        precission: float = 0
):
    """
    Train autoencoder.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    loss_func : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    loader : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    num_epochs : TYPE
        DESCRIPTION.
    precission : float, optional
        DESCRIPTION. The default is 0.001.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    loss_log = {x: [] for x in list(loader.keys())}

    best_model_wts = deepcopy(model.state_dict())
    best_loss = sys.maxsize

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        t0 = time.time()

        model, loss_log = __train_epoch(
            model,
            loss_func,
            device,
            loader,
            optimizer,
            loss_log
        )

        epoch_time = time.time() - t0

        if loss_log['train'][-1] < best_loss:
            best_loss = loss_log['train'][-1]
            best_model_wts = deepcopy(model.state_dict())
            best_epoch = epoch

        if (epoch > 2):
            if abs(loss_log['train'][-2] - loss_log['train'][-1]) < precission:
                break

        print("Epoch elapsed time: {:.4f}s \n".format(epoch_time))

    print('Best val Loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    model.load_state_dict(best_model_wts)
    return model


def __train_epoch(
        model,
        loss_func,
        device,
        loader,
        optimizer,
        loss_log
):
    for phase in list(loader.keys()):
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0

        for batch_idx, data in enumerate(loader[phase]):
            optimizer.zero_grad()
            
            inputs, targets = data
            inputs = inputs.to(device)


            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            print(inputs, outputs, targets)
        
            loss = loss_func(targets.to(device), outputs)

            if phase == 'train':
                
                loss.backward()
                optimizer.step()
                

            running_loss += loss.item()

        epoch_loss = running_loss/len(loader[phase])

        print(f'{phase} Loss:{epoch_loss:.4f}')
        loss_log[phase].append(epoch_loss)

        return model, loss_log
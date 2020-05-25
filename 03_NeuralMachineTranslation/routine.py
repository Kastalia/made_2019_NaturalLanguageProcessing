import torch
import torch.nn as nn
import torch.optim as optim

import time
import math

import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt

from IPython.display import clear_output


def train(model, iterator, optimizer, criterion, clip, show_plots=True, train_history=None, valid_history=None):
    model.train()    
    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if show_plots:
            history.append(loss.cpu().data.numpy())
            if (i+1)%40==0:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

                clear_output(True)
                ax[0].plot(history, label='train loss')
                ax[0].set_xlabel('Batch')
                ax[0].set_title('Train loss')
                if train_history is not None:
                    ax[1].plot(train_history, label='general train history')
                    ax[1].set_xlabel('Epoch')
                if valid_history is not None:
                    ax[1].plot(valid_history, label='general valid history')
                plt.legend()

                plt.show()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):    
    model.eval()    
    epoch_loss = 0    
    history = []    
    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def train_model(model, iterators, optimizer, criterion, scheduler, n_epochs,  clip, show_plots=True, nameModel="model.pth"):
    train_history = []
    valid_history = []

    best_valid_loss = float('inf')
    
    since = time.time()
    for epoch in range(n_epochs):
        train_loss = train(model, iterators["train"], optimizer, criterion, clip, show_plots, train_history, valid_history)
        valid_loss = evaluate(model, iterators["valid"], criterion)
        
        scheduler.step(valid_loss)
        #scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            filename = "{2}loss_{0}_epoch{1}".format(nameModel, epoch, best_valid_loss)
            with open(f"{filename}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print(f'Epoch: {epoch:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_valid_loss))
    return model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

 


def train(model, params, data_loader, loss_fn, epochs = 50, lr = 0.001, 
                      use_scheduler = False, verbose = True):
    
    
    optimizer = torch.optim.Adam(params, lr=lr)
    if use_scheduler: 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 8, 
                                                               cooldown = 0, min_lr = 1e-8, verbose = False)
    for i in tqdm(range(epochs)):
        epoch_loss = 0
        for inputs in data_loader:
            optimizer.zero_grad()
            #inputs = inputs.to(device)                         
            outputs = model(inputs)
            loss = loss_fn(torch.pdist(inputs), torch.pdist(outputs))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose: 
            print(epoch_loss)
        if use_scheduler:
            scheduler.step(epoch_loss)
 
    return epoch_loss

def train_mMDS(n_data, input_dim, out_dim = 2, batch_size = 512, lr = 0.001, epochs = 400, ac_fn = nn.Tanh(), verbose = False,  device = torch.device("cpu")):
    # set network dimensions to d–500–500–2000–out_dim
    enc_1 = nn.Linear(input_dim, 500).to(device)
    enc_2 = nn.Linear(500, 500).to(device)
    enc_3 = nn.Linear(500, 2000).to(device)
    enc_4 = nn.Linear(2000, out_dim).to(device)

    #dropout = nn.Dropout(0.20) # dropout is set to 20 %
    ac_fn = nn.Tanh()
    #ac_fn = nn.ReLU()

    # Train the NN
    data_loader = DataLoader(n_data, shuffle=True, batch_size=batch_size)
    model = lambda x: enc_4(ac_fn(enc_3(ac_fn(enc_2(ac_fn(enc_1(x)))))))
    
    params = list(enc_1.parameters()) + list(enc_2.parameters()) + \
             list(enc_3.parameters()) + list(enc_4.parameters()) 
    
    loss_fn = nn.MSELoss(reduction = 'mean')
    total_loss = train(model, params, data_loader, loss_fn = loss_fn,  
                          epochs = epochs, lr = lr, use_scheduler = True, verbose = verbose)
    
    return total_loss, model 


import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
from wspd import run_wspd, compute_analytics
from tqdm import tqdm
from pathlib import Path

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
 
#device = torch.device("cpu")




def train(model, params, data_loader, loss_fn, epochs = 50, lr = 0.001, 
                      use_scheduler = False, verbose = True):
    
    optimizer = torch.optim.Adam(params, lr=lr)
    if use_scheduler: 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 8, 
                                                               cooldown = 0, min_lr = 1e-8, verbose = True)
    for i in tqdm(range(epochs)):
        epoch_loss = 0
        for inputs in data_loader:
            optimizer.zero_grad()
            #inputs = inputs.to(device)
             
            #outputs = model(inputs)
            loss = loss_fn(model, inputs)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose: 
            print(epoch_loss)
        if use_scheduler:
            scheduler.step(epoch_loss)
 
    return epoch_loss
 
 
def reconstruction_loss(model, inputs):
    outputs = model(inputs)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    return loss_fn(inputs, outputs)
    
    
def adjust_wspd_centers_loss(model, inputs,lcenters_X_wspd, rcenters_X_wspd, n_data, device, dumbells):

    lcenters = lcenters_X_wspd[inputs]
    rcenters = rcenters_X_wspd[inputs]
    batch_dumbells = [dumbells[x] for x in inputs]

    batch_lcenters = model(lcenters)
    batch_rcenters = model(rcenters)
    breakpoint()
    
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = 0
    
    for i in range(len(batch_dumbells)):
        left_dumbell_pts = n_data[batch_dumbells[i][0]]
        right_dumbell_pts = n_data[batch_dumbells[i][1]]
        dumbell_dist = torch.cdist(left_dumbell_pts, right_dumbell_pts, p=2)
        dumbell_apx_dist = torch.empty(dumbell_dist.shape, device=device).fill_(torch.norm(batch_lcenters[i] - batch_rcenters[i], p=2))
        loss += loss_fn(dumbell_dist, dumbell_apx_dist)
"""    
def wspd_loss(model, inputs):
    global lcenters
    global rcenters
    global dumbells
    global n_data
    global device
    batch_lcenters = model(lcenters[inputs].to(device))
    batch_rcenters = model(rcenters[inputs].to(device))
    batch_dumbells = [dumbells[x] for x in inputs]
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = 0
    
    for i in range(len(batch_dumbells)):
        left_dumbell_pts = n_data[batch_dumbells[i][0]]
        right_dumbell_pts = n_data[batch_dumbells[i][1]]
        dumbell_dist = torch.cdist(left_dumbell_pts, right_dumbell_pts, p=2)
        dumbell_apx_dist = torch.empty(dumbell_dist.shape, device=device).fill_(torch.norm(batch_lcenters[i] - batch_rcenters[i], p=2))
        loss += loss_fn(dumbell_dist, dumbell_apx_dist)
        
    return loss
"""    

if __name__ == "__main__":
    #data = pd.read_csv('../data/logistic_100_5.tsv', header = None, sep='\t')
    #data = pd.read_csv('../data/logistic_5000_10.tsv', header = None, sep='\t')
    #data = pd.read_csv('../data/logistic_10000_20.tsv', header = None, sep='\t')
    #data = pd.read_csv('../data/gumbel_5000_10.tsv', header = None, sep='\t')
    data, file_name = pd.read_csv('../data/laplace_10000_20.tsv', header = None, sep='\t'), 'laplace_10000_20'
    #data = pd.read_csv('../../data4Domagoj/muraro-prepare-log_count.csv', header = None)
    #data = pd.read_csv('../../data4Domagoj/mouseCNSzeisel_pca100.csv', header = None)
    #data = pd.read_csv('../../data4Domagoj/jurkat_pca100.tsv', header = None, sep='\t').drop([0], axis=1)
    #data = pd.read_csv('../../data4Domagoj/pbmc_pca100.tsv', header = None, sep='\t').drop([0], axis=1)
    
    
    
    
    
    input_dim = data.shape[1]
    input_size = data.shape[0]
    print("input_dim = ", input_dim, " and input_size = ", input_size)
    data = torch.tensor(data.values, dtype=torch.float32)
    
    NormalizeData = lambda x: x # don't normalize
    
    n_data = NormalizeData(data).to(device)
    
    
    # set network dimensions to d–500–500–2000–10
    enc_1 = nn.Linear(input_dim, 500).to(device)
    enc_2 = nn.Linear(500, 500).to(device)
    enc_3 = nn.Linear(500, 200).to(device)
    enc_4 = nn.Linear(200, 3).to(device)

    dec_1 = nn.Linear(500, input_dim).to(device)
    dec_2 = nn.Linear(500, 500).to(device)
    dec_3 = nn.Linear(200, 500).to(device)
    dec_4 = nn.Linear(3, 200).to(device)

    dropout = nn.Dropout(0.20) # dropout is set to 20 %
    #ac_fn = nn.Tanh()
    ac_fn = nn.ReLU()
    
    """
    # Train the first autoencoder
    data_loader = DataLoader(n_data, shuffle=True, batch_size=256)
    params = list(enc_1.parameters()) + list(dec_1.parameters())
    model = lambda x: dec_1(ac_fn(enc_1(dropout(x))))
    l = train(model, params, data_loader, loss_fn = reconstruction_loss, 
                          epochs = 100, lr = 0.001, use_scheduler = True, verbose = False)
    print("autoencoder 1 loss: ", l)
    
    
    # Train the second autoencoder
    data_loader = DataLoader(ac_fn(enc_1(n_data)).detach(), shuffle=True, batch_size=256)
    params = list(enc_2.parameters()) + list(dec_2.parameters())
    model = lambda x: dec_2(ac_fn(enc_2(dropout(x))))
    l = train(model, params, data_loader, loss_fn = reconstruction_loss, 
                          epochs = 100, lr = 0.001, use_scheduler = True, verbose = False)
    print("autoencoder 2 loss: ", l)
    
    # Train the third autoencoder
    data_loader = DataLoader(ac_fn(enc_2(ac_fn(enc_1(n_data)))).detach(), shuffle=True, batch_size=256)
    params = list(enc_3.parameters()) + list(dec_3.parameters())
    model = lambda x: dec_3(ac_fn(enc_3(dropout(x))))
    l = train(model, params, data_loader, loss_fn = reconstruction_loss,  
                          epochs = 100, lr = 0.001, use_scheduler = True, verbose = False)
    print("autoencoder 3 loss: ", l)
    
    # Train the fourth autoencoder
    data_loader = DataLoader(ac_fn(enc_3(ac_fn(enc_2(ac_fn(enc_1(n_data)))))).detach(), shuffle=True, batch_size=256)
    params = list(enc_4.parameters()) + list(dec_4.parameters())
    model = lambda x: dec_4(ac_fn(enc_4(dropout(x))))
    l = train(model, params, data_loader, loss_fn = reconstruction_loss,  
                          epochs = 100, lr = 0.001, use_scheduler = True, verbose = False)
    print("autoencoder 4 loss: ", l)
    """
    

    # Train the complete autoencoder 
    data_loader = DataLoader(n_data, shuffle=True, batch_size=512)
    encoder = lambda x: enc_4(ac_fn(enc_3(ac_fn(enc_2(ac_fn(enc_1(x)))))))
    decoder = lambda x: dec_1(ac_fn(dec_2(ac_fn(dec_3(ac_fn(dec_4(x)))))))
    model = lambda x: decoder(ac_fn(encoder(x)))
    params = list(enc_1.parameters()) + list(enc_2.parameters()) + \
             list(enc_3.parameters()) + list(enc_4.parameters()) + \
             list(dec_4.parameters()) + list(dec_3.parameters()) + \
             list(dec_2.parameters())+ list(dec_1.parameters())

    l = train(model, params, data_loader, loss_fn = reconstruction_loss,  
                          epochs = 500, lr = 0.001, use_scheduler = True, verbose = True)
    print("complete autoencoder loss: ", l)
    
    # compute WSPD 
    (lcenters, rcenters, dumbells) = run_wspd(n_data, encoder(n_data).detach(), verbose = True)

    """
    # remap dumbbell centers to new coordinates  
    # dataloader not contains only indices. Loss function will take care of the rest
    
    data_loader = DataLoader(torch.tensor([x for x in range(len(lcenters)) if len(dumbells[x][0])> 1 or len(dumbells[x][1])> 1]),
                             shuffle=False, batch_size=64)
                                
    print("Size of dataloader: ", len(data_loader))
    model = lambda x: decoder(ac_fn(x))
    params = list(dec_4.parameters()) + list(dec_3.parameters()) + \
             list(dec_2.parameters())+ list(dec_1.parameters())
    l = train(model, params, data_loader, loss_fn = wspd_loss,  
                          epochs = 10, lr = 0.001, use_scheduler = False, verbose = True)
    """
    
    # copy the model back to cpu
    (enc_1.cpu(), enc_2.cpu(), enc_3.cpu(), enc_4.cpu(), dec_1.cpu(), dec_2.cpu(), dec_3.cpu(), dec_4.cpu())
    # run analytics
    compute_analytics(dumbells, lcenters, rcenters, n_data, encoder, decoder, device, ac_fn, verbose = False)
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def reconstruction_loss(model, inputs):
    outputs = model(inputs)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    return loss_fn(inputs, outputs)


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

def train_autoencoder(n_data, input_dim, out_dim = 2, batch_size = 512, lr = 0.001, epochs = 400, ac_fn = nn.Tanh(), verbose = False,  device = torch.device("cpu")):
    # set network dimensions to d–500–500–2000–out_dim
    enc_1 = nn.Linear(input_dim, 500).to(device)
    enc_2 = nn.Linear(500, 500).to(device)
    enc_3 = nn.Linear(500, 2000).to(device)
    enc_4 = nn.Linear(2000, out_dim).to(device)

    dec_1 = nn.Linear(500, input_dim).to(device)
    dec_2 = nn.Linear(500, 500).to(device)
    dec_3 = nn.Linear(2000, 500).to(device)
    dec_4 = nn.Linear(out_dim, 2000).to(device)

    #dropout = nn.Dropout(0.20) # dropout is set to 20 %


    # Train the complete autoencoder 
    #(enc_1.to(device), enc_2.to(device), enc_3.to(device), enc_4.to(device), dec_1.to(device), dec_2.to(device), dec_3.to(device), dec_4.to(device))
    data_loader = DataLoader(n_data, shuffle=True, batch_size=batch_size)
    encoder = lambda x: enc_4(ac_fn(enc_3(ac_fn(enc_2(ac_fn(enc_1(x)))))))
    decoder = lambda x: dec_1(ac_fn(dec_2(ac_fn(dec_3(ac_fn(dec_4(x)))))))
    model = lambda x: decoder(ac_fn(encoder(x)))
    params = list(enc_1.parameters()) + list(enc_2.parameters()) + \
             list(enc_3.parameters()) + list(enc_4.parameters()) + \
             list(dec_4.parameters()) + list(dec_3.parameters()) + \
             list(dec_2.parameters())+ list(dec_1.parameters())

    total_loss = train(model, params, data_loader, loss_fn = reconstruction_loss,  
                          epochs = epochs, lr = lr, use_scheduler = True, verbose = verbose)
    
    return total_loss, encoder, decoder


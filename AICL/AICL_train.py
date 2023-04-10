import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.audio_data import ESC50, SpeechCommands, AVE
from models.visual_model import ASTModel
from models.pet_modules import ConvPass

"""
Arguments Parser
"""
def parse_options():
    parser = argparse.ArgumentParser(description="Adapter TICL")
    parser.add_argument('--gpu_id', type=str, default="cuda:0", help='the gpu id')
    opts = parser.parse_args()
    torch.manual_seed(1111)
    opts.device = torch.device(opts.gpu_id)
    return opts


"""
Collate Function
"""
def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    spectrograms,labels = [], []
    # Gather in lists, and encode labels as indices
    for spec,label in batch:
        spectrograms += [spec]
        labels += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    spectrograms = af_pad_sequence(spectrograms)
    labels = torch.stack(labels)
    return spectrograms,labels


"""
Train and Eval Functions
"""
def train_one_epoch(train_data_loader,model,optimizer,loss_fn,device):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0  
    model.train()
    ###Iterating over data loader
    for data, labels in train_data_loader:       
        #Loading data and labels to device
        data = data.to(device)
        labels = labels.to(device)       
        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(data)
        #Calculating Loss
        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())       
        #Backward
        _loss.backward()
        optimizer.step()

        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)
    acc = round(sum_correct_pred/total_samples,4)*100   
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)   
    return epoch_loss, acc

def val_one_epoch(val_data_loader, model,loss_fn,device):   
    ### Local Parameters
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
      for data, labels in val_data_loader:
        #Loading data and labels to device
        data = data.to(device)
        labels = labels.to(device)
        #Forward
        preds = model(data)       
        #Calculating Loss
        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())       
        # print(torch.argmax(preds,dim=1),labels)
        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)
    acc = round(sum_correct_pred/total_samples,4)*100    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)        
    return epoch_loss, acc


"""
Save weights - In Adapter Incremental setting - only pos embed, classifier weights and convolutional adapter weights need to be saved. 
"""
def save_model_weights(model, wt_name, save_mode):
    if save_mode == 'mlp':
        torch.save(model.mlp_head.state_dict(),wt_name)
    elif save_mode == 'pos':
        torch.save(model.v.pos_embed,wt_name)
    elif save_mode == 'adapter':
        weights = model.state_dict()
        # saving the adapter weights only!
        for name in weights:
            # isolating the convolutional adapter weights 
            if name.split('.')[1] == 'blocks':
                if name.split('.')[3] in ['conv1','conv2']:
                    # print(name)
                    continue
                else:
                    weights[name] = torch.zeros(1) # zero the non-adapter weights within encoder (reduce storage!)
            else:
                weights[name] = torch.zeros(1) # zero the non encoder layer weights (reduce storage!)
        torch.save(weights,wt_name) # 5.6 MB storage (full model, say ESC50, requires 336 MB storage)

"""
load adapter weights only
"""
def load_adapter_weights(current_model, target_wt):

    current_wts = current_model.state_dict()

    for name in current_wts:
        # replacing convolutional adapter weights 
        if name.split('.')[1] == 'blocks':
            if name.split('.')[3] in ['conv1','conv2']:
                current_wts[name] = target_wt[name]
            else:
                continue # don't disturb other weights
        else:
            continue # don't disturb other weights
    current_model.load_state_dict(current_wts)
    return current_model


"""
Interpolate Pos Embed
"""
def interpolate_pos_embed(f_dim, t_dim, pos_embed):

  original_num_patches = 576 # 1 + 24*24 tokens
  original_embedding_dim =768
  oringal_hw = 24
  num_patches = f_dim * t_dim

  # get the positional embedding from vit model, skip the first cls tokens, reshape it to original 2D shape (24*24).
  new_pos_embed = pos_embed[:, 1:, :].detach().reshape(1, original_num_patches, original_embedding_dim).transpose(1, 2).reshape(1, original_embedding_dim, oringal_hw, oringal_hw)

  # cut (from middle) or interpolate the second dimension of the positional embedding
  if t_dim <= oringal_hw:
      new_pos_embed = new_pos_embed[:, :, :, int(oringal_hw / 2) - int(t_dim / 2): int(oringal_hw / 2) - int(t_dim / 2) + t_dim]
  else:
      new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(oringal_hw, t_dim), mode='bilinear')

  # cut (from middle) or interpolate the first dimension of the positional embedding
  if f_dim <= oringal_hw:
      new_pos_embed = new_pos_embed[:, :, int(oringal_hw / 2) - int(f_dim / 2): int(oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
  else:
      new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

  # flatten the positional embedding
  new_pos_embed = new_pos_embed.reshape(1, original_embedding_dim, num_patches).transpose(1,2)
  # concatenate the above positional embedding with the cls token of the deit model.
  pos_embed = nn.Parameter(torch.cat([pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))

  return pos_embed


"""
Main Loop
"""
def train_test(args):

    #####################################################################################################################################
    # Google SpeechCommandsV2 dataset
    #####################################################################################################################################
    print("\n\nCurrent Dataset - SpeechCommandsV2")
    # DataLoader
    train_dataset = SpeechCommands(split='train')
    test_dataset = SpeechCommands(split='test')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=16)
    sc_test_loader = test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=16)
    print("\t Dataset Loaded")

 
    model = ASTModel(label_dim=35, input_fdim=128, input_tdim=101, unsqueeze=True)
    sc_fdim, sc_tdim = model.get_shape(fstride=10,tstride=10,input_fdim=128,input_tdim=101)
    model.to(args.device)
    print("\t Model Loaded")

    Adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad) - sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    Classifier_params = sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    print('\n\t Frozen Backbone params = ',sum(p.numel() for p in model.parameters()) - Adapter_params - Classifier_params)
    print('\n\t SC Adapter params = ',Adapter_params)
    print('\t SC Classifier params = ',Classifier_params)
    # Total current parameters = Frozen BB params + Speech Commands (Adapter + Classifier) params

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []
    print("\t Started Training")
    for epoch in range(5):          
        ###Training
        loss, acc = train_one_epoch(train_loader,model,optimizer,loss_fn,args.device)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn,args.device)
        best_val_acc.append(val_acc)
    
    """
    Eval 1
    """
    print("\n\t Acc of Speech Commands V2 dataset.....", round(np.max(np.asarray(best_val_acc)),2))
    save_model_weights(model,'WEIGHTS/adapter_incremental/sc_classifier.pth',save_mode='mlp')
    save_model_weights(model,'WEIGHTS/adapter_incremental/sc_pos_embed.pth',save_mode='pos')
    save_model_weights(model,'WEIGHTS/adapter_incremental/sc_adapter.pth',save_mode='adapter')

    
    #####################################################################################################################################
    # ESC50 dataset
    #####################################################################################################################################
    print("\n\nCurrent Dataset - ESC50")
    # DataLoader
    train_anno = "/home/nithish/Audio_PET/ESC50/protocols/train4.csv"
    test_anno = "/home/nithish/Audio_PET/ESC50/protocols/test4.csv"
    train_dataset = ESC50(train_anno, "/home/nithish/Audio_PET/ESC50/spectrograms/")
    test_dataset = ESC50(test_anno, "/home/nithish/Audio_PET/ESC50/spectrograms/")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    esc50_test_loader = test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)
    print("\t Dataset Loaded")

     
    # first find the shape of patches f_dim and t_dim for ESC50 dataset. Utilize model.get_shape() for this
    esc50_fdim, esc50_tdim = model.get_shape(fstride=10,tstride=10,input_fdim=128,input_tdim=501)
    # replace adapters (the backbone still remains the same)
    for i in range(12): model.v.blocks[i] = ConvPass(model.v.blocks[i], 32, esc50_fdim, esc50_tdim)
    # replace classifier
    model.mlp_head = nn.Linear(768,50)
    # interpolate original timm pos-embed for Speech Commands
    esc50_pos_embed = interpolate_pos_embed(f_dim=esc50_fdim, t_dim=esc50_tdim, pos_embed=torch.load('original_ViT_IN21k_pretrained_pos_embed.pth'))
    esc50_pos_embed.requires_grad=False
    model.v.pos_embed = esc50_pos_embed
    # Since the model's backbone weights are frozen, they are unaffected, no matter whichever task you train!
    model.unsqueeze=False
    
    model.to(args.device)
    print("\t Model Loaded")
    Adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad) - sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    Classifier_params = sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    print('\t ESC 50 Adapter params = ',Adapter_params)
    print('\t ESC50 Classifier params = ',Classifier_params)
    # Total current parameters = Frozen BB params + SC (Adapter + Classifier) params + ESC 50 (Adapter + Classifier) params

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []
    print("\n\t Started Training")
    for epoch in range(20):          
        ###Training
        loss, acc = train_one_epoch(train_loader,model,optimizer,loss_fn,args.device)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn,args.device)
        best_val_acc.append(val_acc)

    """
    Eval 2
    """
    print("\n\t Acc of ESC-50 dataset.....", round(np.max(np.asarray(best_val_acc)),2))
    # Save ESC50 task specific parameters
    save_model_weights(model,'WEIGHTS/adapter_incremental/esc50_classifier.pth',save_mode='mlp')
    save_model_weights(model,'WEIGHTS/adapter_incremental/esc50_pos_embed.pth',save_mode='pos')
    save_model_weights(model,'WEIGHTS/adapter_incremental/esc50_adapter.pth',save_mode='adapter') 

    #  Inference on SpeechCommands
    model.v.pos_embed = torch.load('WEIGHTS/adapter_incremental/sc_pos_embed.pth')
    model.mlp_head = nn.Linear(768,35)
    model.mlp_head.load_state_dict(torch.load('WEIGHTS/adapter_incremental/sc_classifier.pth'))
    # note that the attn weights and proj weights remain the same. 
    for i in range(12): model.v.blocks[i] = ConvPass(model.v.blocks[i], 32, sc_fdim, sc_tdim)
    model = load_adapter_weights(model,torch.load('WEIGHTS/adapter_incremental/sc_adapter.pth'))

    model.unsqueeze=True
    model.f_dim,model.t_dim = sc_fdim, sc_tdim
    
    model.to(args.device)
    _, sc_acc = val_one_epoch(sc_test_loader, model, loss_fn, args.device)
    print("\n\t Acc on SpeechCommandsV2 after training on ESC50.....", sc_acc)

    
    #####################################################################################################################################
    # Audio Visual Event (audio only) Dataset
    #####################################################################################################################################
    print("\n\nCurrent Dataset - AVE")
    # DataLoader
    train_anno = "/home/DSO/DSO_server/AVE/train.csv"
    test_anno = "/home/DSO/DSO_server/AVE/test.csv"
    train_dataset = AVE(train_anno, "/home/DSO/DSO_server/AVE/audio_files/")
    test_dataset = AVE(test_anno, "/home/DSO/DSO_server/AVE/audio_files/")
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_fn, num_workers=16)
    ave_test_loader = test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, collate_fn=collate_fn, num_workers=16)
    print("\t Dataset Loaded")

    # first find the shape of patches f_dim and t_dim for AVE dataset. Utilize model.get_shape() for this
    ave_fdim, ave_tdim = model.get_shape(fstride=10,tstride=10,input_fdim=128,input_tdim=1006)
    # replace adapters
    for i in range(12): model.v.blocks[i] = ConvPass(model.v.blocks[i], 32, ave_fdim, ave_tdim)
    # replace classifier
    model.mlp_head = nn.Linear(768,28)
    # interpolate original timm pos-embed for Speech Commands
    ave_pos_embed = interpolate_pos_embed(f_dim=ave_fdim, t_dim=ave_tdim, pos_embed=torch.load('original_ViT_IN21k_pretrained_pos_embed.pth'))
    ave_pos_embed.requires_grad=False
    model.v.pos_embed = ave_pos_embed
    # Since the model's backbone weights are frozen, they are unaffected, no matter whichever task you train!
    model.unsqueeze=True # required for AVE because of the spectrogram shape!
    
    model.to(args.device)
    print("\t Model Loaded")
    Adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad) - sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    Classifier_params = sum(p.numel() for p in model.mlp_head.parameters() if p.requires_grad)
    print('\n\t AVE Adapter params = ',Adapter_params)
    print('\t AVE Classifier params = ',Classifier_params)
    # Total current parameters = Frozen BB params + ESC 50 (Adapter + Classifier) params + Speech Commands (Adapter + Classifier) params + AVE (Adapter + Classifier) params

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []
    print("\n\t Started Training")
    for epoch in range(15):          
        ###Training
        loss, acc = train_one_epoch(train_loader,model,optimizer,loss_fn,args.device)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn,args.device)
        best_val_acc.append(val_acc)
    
    """
    Eval 3
    """
    print("\n\t Acc of AVE dataset.....", round(np.max(np.asarray(best_val_acc)),2))
    save_model_weights(model,'WEIGHTS/adapter_incremental/ave_classifier.pth',save_mode='mlp')
    save_model_weights(model,'WEIGHTS/adapter_incremental/ave_pos_embed.pth',save_mode='pos')
    save_model_weights(model,'WEIGHTS/adapter_incremental/ave_adapter.pth',save_mode='adapter')

    #  Inference on SpeechCommands
    model.v.pos_embed = torch.load('WEIGHTS/adapter_incremental/sc_pos_embed.pth')
    model.mlp_head = nn.Linear(768,35)
    model.mlp_head.load_state_dict(torch.load('WEIGHTS/adapter_incremental/sc_classifier.pth'))
    # note that the attn weights and proj weights remain the same. 
    for i in range(12): model.v.blocks[i] = ConvPass(model.v.blocks[i], 32, sc_fdim, sc_tdim)
    model = load_adapter_weights(model,torch.load('WEIGHTS/adapter_incremental/sc_adapter.pth'))

    model.unsqueeze=True
    model.f_dim,model.t_dim = sc_fdim, sc_tdim
    
    model.to(args.device)
    _, sc_acc = val_one_epoch(sc_test_loader, model, loss_fn, args.device)
    print("\n\t Acc on SpeechCommandsV2 after training on AVE.....", sc_acc)

    
    #  Inference on ESC50
    model.v.pos_embed = torch.load('WEIGHTS/adapter_incremental/esc50_pos_embed.pth')
    model.mlp_head = nn.Linear(768,50) # classifer
    model.mlp_head.load_state_dict(torch.load('WEIGHTS/adapter_incremental/esc50_classifier.pth'))
    # note that the attn weights and proj weights remain the same. 
    for i in range(12): model.v.blocks[i] = ConvPass(model.v.blocks[i], 32, esc50_fdim, esc50_tdim)
    model = load_adapter_weights(model,torch.load('WEIGHTS/adapter_incremental/esc50_adapter.pth'))

    model.unsqueeze=False
    model.f_dim,model.t_dim = esc50_fdim, esc50_tdim

    model.to(args.device)
    _, esc50_acc = val_one_epoch(esc50_test_loader, model, loss_fn, args.device)
    print("\n\t Acc on ESC-50 after training on AVE.....", esc50_acc)

if __name__ == "__main__":
    opts = parse_options()
    train_test(args=opts)
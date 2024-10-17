import os
import numpy as np
import time
import sys
import cv2
from torchvision import transforms
from PIL import Image




import torch
import torchvision.transforms as transforms
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, output, target,padding_mask):
        # Apply log-softmax directly to the output for numerical stability
        log_softmax_output = F.log_softmax(output, dim=-1)
       
        # Calculate the loss
        # If target is one-hot encoded, use:
        
        loss = -torch.sum(target * log_softmax_output, dim=-1)

        # Mean over the batch
        loss = loss *padding_mask
        return loss.sum() / padding_mask.sum()
    
class Trainer:





    def train (model, train_dataset,batch_size, trMaxEpoch, start_token,vocab_size,opt=None, checkpoint=None,transResize = 224,RGB=False):

        
        transCrop =224
        
        model = torch.nn.DataParallel(model).cuda()

        # Data Transformer
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.Resize(transResize))
        # transformList.append(transforms.RandomResizedCrop(transCrop))
        # transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList) 
                
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        if opt is None:
            optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        else:
            optimizer = opt
        # scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        # loss = torch.nn.BCELoss(size_average = True)
        loss = CustomCrossEntropyLoss()
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            loss_value = 0
            model.train()
            for step in range(0,len(train_dataset[1])//batch_size):

                # Define the encoder and decoder inputs, and the decoder output

                ### resize images and transform them into pytorch tensor
                encoder_input = []
                imgs = train_dataset[0].select(  range( step*batch_size,(step+1)*batch_size ))['image']
                for img in imgs:
                    if RGB:
                        img = img.convert("RGB")
                    transImg = transformSequence(img).to('cuda')
                    encoder_input.append(transImg)
                encoder_input = torch.stack(encoder_input,dim=0)

                # Get Batch Target
                target = train_dataset[1][step*batch_size:(step+1)*batch_size, :]
            
                
                # Train Step
                loss_value+= Trainer.step(model, encoder_input, target, optimizer,loss,vocab_size,start_token,step)

            
                if step % 50 ==0:
                    print (f'Epoch [ {str(epochID + 1)} step-{step} ] loss= {str(loss_value/(step+1))} ')
                
            # lossVal, losstensor = Trainer.epochVal (model, val_data, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            print (f'Epoch [ {str(epochID + 1)} ] total_loss= {str(loss_value/(step+1))} \n')
            
            # timestampTime = time.strftime("%H%M%S")
            # timestampDate = time.strftime("%d%m%Y")
            # timestampEND = timestampDate + '-' + timestampTime
            
            # scheduler.step(losstensor.data[0])
            
            # if lossVal < lossMIN:
            #     lossMIN = lossVal    
            #     # torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
            #     print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            # else:
            #     print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
        return model,optimizer,loss_value,epochID
                     
    #-------------------------------------------------------------------------------- 

    def step (model, input, target, optimizer,loss,vocab_size,start_token,step=-1):
        
        optimizer.zero_grad()
        
        
                    
        # print(target.device)
        target = target.cuda(torch.device('cuda')).long()
        captions = target[:,:-1]
        target_for_loss = target[:,1:]


        padding_mask = target_for_loss != 0
        caption_mask = Trainer.make_target_mask(captions)
        # captions = torch.zeros(target.shape,dtype=torch.int64).to('cuda')
        # # captions[:,0] = start_token   
        # # if step ==0:
        # #     print(captions)
        varInput = torch.autograd.Variable(input).to('cuda')
        varCaptions = torch.autograd.Variable(captions)
        varCaptionsMask = torch.autograd.Variable(caption_mask).to('cuda')

        varOutput = model(varInput,varCaptions,varCaptionsMask)
        # print(varTarget.size())
        # target = torch.nn.functional.one_hot(target.to(torch.int64))

        target_for_loss = torch.nn.functional.one_hot(target_for_loss,vocab_size)

        varTarget = torch.autograd.Variable(target_for_loss)      
        lossvalue = loss(varOutput.to('cuda'), varTarget,padding_mask)
        # lossvalue = lossvalue.to('cuda')
        
        # print(captions.device)
        # print(varInput.device)
        # print(varCaptions.device)
        # print(varOutput.device)
        # print(lossvalue.device)
                
        lossvalue.backward()
        optimizer.step()
        return lossvalue
            
    #-------------------------------------------------------------------------------- 



    def epochVal (model, dataLoader, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        
        for i, (input, target) in enumerate (dataLoader):
            
            target = target.cuda(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                 
            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)    
            varOutput = model(varInput)
            
            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor
            
            lossVal += losstensor.data[0]
            lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
               
    #-------------------------------------------------------------------------------- 

    def make_target_mask( target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask
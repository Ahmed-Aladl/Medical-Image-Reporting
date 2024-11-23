import streamlit as st
from PIL import Image
import torch
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import torchvision.transforms as transforms
import json

import models
from models import EncoderCNN,Decoder
import cv2


####################################_____________________#############################
model_ckpt = "pathTo/model.pth"                                                         ###
toknizer_json_path = "pathTo/tokenizer.json"                                            ###
####################################_____________________#############################


feature_dim = 8
h, n= 8, 6
d_model = 512

v_in_dim,v_out_dim, d_ff= 512 ,256 , d_model*4
feature_projection = False

if feature_projection is not True:
  v_in_dim = feature_dim**2
vocab_size = 8539
token_len = sequence_len = 186

dropout_rate = 0.3
img_reshape_size = (feature_dim*32,feature_dim*32)
hidden_size = 512
batch_size = 8
epochs = 1



### Generate Lookahead Mask
def make_target_mask( target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask

############################___________________________###################$############
############################___________________________###################$############
############################___________________________###################$############
def generate_report(model,img,token_len,start_eos):
    model.eval().to('cpu')

    output = torch.zeros(   (1, token_len-1)     ).long().to("cpu")
    output[:,0] = start_eos['start']

    mask = make_target_mask(output)
    with torch.no_grad():
        memory = model.encoder(img)
    for i in range(0, output.shape[-1] ):

        if output[0,i] != start_eos['eos']:

            with torch.no_grad():
            # Decoder && prediction forward pass
                decoder_output = model.decoder(output, memory, None,mask)
                predictions = model.final_linear(decoder_output)[0]
            word_index = torch.argmax(   torch.nn.functional.softmax(predictions[i],dim=-1)  )
            
            if i == token_len-2:
                 break
            output[0,i+1] = word_index
        else:
            break
    return output



######################______________________________________#######################
# Data Transformer
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transformList = []
transformList.append(transforms.Resize((256,256)))
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList) 

def preprocess_img(imgs):
    global transformSequence
    processed_imgs = []
    for img in imgs:
        transImg = transformSequence(img).to('cpu')
        processed_imgs.append(transImg)

    processed_imgs = torch.stack(processed_imgs,dim=0)
    
    return processed_imgs

######################______________________________________#######################
## Convert sequences back to text
def sequences_to_texts(sequences,tokenizer):
    texts = []
    for sequence in sequences:
        sentence= ""
        for word in sequence:
            if int(word)==0:
                break
            sentence+= tokenizer.index_word[int(word)] + " "
        texts.append(sentence)
    return texts





############################___________________________###################$############



### Load Tokenizer Form Json File
def load_tokenizer(file = toknizer_json_path):
    with open(file, 'r') as json_file:
        tokenizer_json = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

#########################################################################
#                                                                       #
#                                                                       #    
#########################################################################

# Recreate the tokenizer
tokenizer = load_tokenizer()

start_eos={
      'start':tokenizer.word_index['start'],
      'eos':tokenizer.word_index['eos']
    }




model_ckpt = torch.load(model_ckpt,map_location=torch.device('cpu'))
import numpy as np


# build and load pre-trained model
encoder = EncoderCNN(v_in_dim,feature_dim**2,weights = None,projection = feature_projection,freeze_chexnet=False)
model = models.ImageCaptioningModel(encoder,
                                             vocab_size,
                                             token_len,
                                             d_model,
                                             h,n,
                                             ff_dim= d_ff,
                                             v_in_dim=v_in_dim,
                                             v_out_dim=v_out_dim,
                                             dropout = dropout_rate)
new_state_dict = {}
for k, v in model_ckpt['state_dict'].items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)



st.title("Model Inference in Streamlit")
# Input text
user_input = st.text_input("Enter text for inference:", "")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])



if st.button("Run Inference"):
    
    # st.write(np.array(image).shape)
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=False)
                
        # Run the model inference
        img = preprocess_img([image])
        report =generate_report(model,img,sequence_len,start_eos)
        text = sequences_to_texts(report,tokenizer)
        ###         
        st.write("Output:")
        st.write(text)
        # st.write(user_input)
    else:
        st.write("Please upload an image.")
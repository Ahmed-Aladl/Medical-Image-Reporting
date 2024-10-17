import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torchvision


# Embedding the input sequence
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        # self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        # x = self.dropout(x)
        return x

# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, value)

        return output
        
# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size,-1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size,  -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output
        
# Multi-head Cross-attention layer 
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads,v_in_dim = 64 ,v_out_dim=128 , dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(v_in_dim, embedding_dim)
        self.value_projection = nn.Linear(v_in_dim, v_out_dim)

        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(v_out_dim, embedding_dim)
        self.v_out_dim = v_out_dim
    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        sequence_len = value.size(1)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, sequence_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_len, self.num_heads, -1).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.v_out_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer decoder layer
class OldDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x
    




class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048,v_in_dim=64, v_out_dim = 128, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(embedding_dim, num_heads,dropout=dropout)
        self.encoder_attention = MultiHeadCrossAttention(embedding_dim, num_heads, v_in_dim=v_in_dim, v_out_dim=v_out_dim,dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, x, memory, source_mask, target_mask):
        # Masked-Multihead Attention        (self attention)
        x = x + self.dropout1(self.self_attention(x, x, x, target_mask))
        # First Norm
        norm = self.norm1(x)
        
        # Cross Attention
        x = norm + self.dropout2(self.encoder_attention(norm, memory, memory, source_mask))
        # Second Norm
        norm = self.norm2(x)
        
        # FeedForward Network
        x = norm + self.dropout3(self.feed_forward(norm))
        # Third Norm
        norm = self.norm3(x)
        return norm

# Decoder transformer
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len,num_heads, num_layers,ff_dim = 2048,v_in_dim=64,v_out_dim =128, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, ff_dim,v_in_dim , v_out_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, target, memory, source_mask, target_mask):
        # Embed the source
        x = self.embedding(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder,target_vocab_size, target_max_seq_len, embedding_dim, num_heads, num_layers,ff_dim = 2048,v_in_dim=64,v_out_dim=128, dropout=0.1,classifier_ff=8):
        super(ImageCaptioningModel, self).__init__()
        
        self.target_vocab_size = target_vocab_size
        self.target_max_seq_len = target_max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
    
        self.encoder = encoder
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, num_heads, num_layers, ff_dim=ff_dim ,v_in_dim=v_in_dim ,v_out_dim=v_out_dim,dropout=dropout)
        self.final_linear = nn.Linear(embedding_dim, target_vocab_size)
        # self.dropout = nn.Dropout(dropout)
        
        
        
        # self.ff = nn.Linear(v_in_dim, classifier_ff),
        # self.relu = nn.ReLU(),
 
        # self.classifier = nn.Linear(classifier_ff*1024, 13)
    
    def forward(self, source, target, target_mask):
        # Encoder forward pass
        memory = self.encoder(source)
        # Decoder forward pass
        output = self.decoder(target, memory, None,target_mask)
        # Final linear layer
        output = self.final_linear(output)


        # #Classification 
        # ff_out = self.ff(memory)
        # classifier_input = self.relu(ff_out)
        # classes =self.classifier(classifier_input)

        return output
    
    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)

    def make_target_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask
    






class EncoderCNN(nn.Module):
    def __init__(self,embed_size,dnesnet_out_features=7*7, weights=None,projection=True,freeze_chexnet=True):
        super(EncoderCNN, self).__init__()
        self.freeze_chexnet = freeze_chexnet
        self.projection = projection
        model = torchvision.models.densenet121(pretrained=True)        
        self.densenet121 = nn.Sequential( *list(     model.children()    )[0] )
        if weights is not None:
            self.densenet121.load_state_dict(torch.load(weights))
        hidden_size= embed_size//2
        # self.hidden_projection = nn.Linear(dnesnet_out_features, hidden_size)  # Embed the features to embed_size
        if projection is True:
            self.fc = nn.Linear(dnesnet_out_features, embed_size)  # Embed the features to embed_size
            self.norm = torch.nn.BatchNorm1d(1024)
            self.relu = nn.ReLU()
        for param in self.densenet121.parameters():
            param.requires_grad = not freeze_chexnet

    def forward(self, images):
        if self.freeze_chexnet:
            with torch.no_grad():  # Disable gradient computation for the pre-trained encoder
                features = self.densenet121(images)  # Extract features from the image
        else:
                features = self.densenet121(images)  # Extract features from the image

        features = features.view(features.size(0),features.size(1), -1)  # view out --> batch_size ,num_of_kernels ,singel_feature_size
        
        # features = self.hidden_projection(features)
        if self.projection is True:
          features = self.fc(features)  # Project to the 512,10
        # features = self.norm(features)
        # features = self.relu(features)  # Apply ReLU activation

        return features
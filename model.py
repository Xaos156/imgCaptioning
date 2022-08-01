#using flickr 8k and 30k datasets
import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary
from random import randint
import cv2
from skimage import io, transform
import math
from torch.autograd import Variable
import random

class captionDataset(torch.utils.data.Dataset):
    vocab = []
    char_to_int = {}
    int_to_char = {}
    
    def __init__(self, root="data/images/", annotations="data/captions.txt", img_size=299, word_level=False):
        self.root = root
        self.IMG_SIZE = img_size
        self.vocab = ["<PAD>", "<START>", "<END>", "\xa0"]
        
        with open(annotations, encoding='utf8') as file:
            csv_reader = np.array(list(csv.reader(file, delimiter="|")))
        files, text = np.split(csv_reader[1:], 2, axis = 1)
        files = files.squeeze(1) #remove the leftover dimension from the split
        text = text.squeeze(1)
        
        if not word_level: #character level tokenization
            self.vocab += sorted(set(str(list(text))))
            self.maxlen = len(max(text, key=len))
        else: #word level tokenization
            self.maxlen = 0
            for i in text:
                for word in i.split(" "):
                    if word not in self.vocab:
                        self.vocab += [word]
                if len(i.split(" ")) > self.maxlen:
                    self.maxlen = len(i.split(" "))
                    
        self.maxlen += 2 # two more because of start and end tokens
                        
        self.char_to_int = dict((c, i) for i, c in enumerate(self.vocab))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.vocab))
        
        #print(self.int_to_char)
        
        captions = []
        for k in text:
            captions += [[1]] # 1 represents <START>
            
            captions[-1] += [self.char_to_int[i] for i in (k.split(" ") if word_level else k)]
            captions[-1] += [2] + [0]*(self.maxlen-len(captions[-1])-1) # 2 represents <END> and 0 is <PAD>
            
            
        
        self.X = files
        self.y = captions
    
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = io.imread(self.root +self.X[idx]) #using skimiage because its RGB and way faster than pillow 
        img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE)).reshape(-1, self.IMG_SIZE, self.IMG_SIZE)/255.0
        caption = np.array(self.y[idx])
        return img.astype(np.float32), caption.astype(np.int64) #cast as float and long numpy arrays  

class encoderCNN(nn.Module):
    def __init__(self, embedding_dim):
        super(encoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv1_bn=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv2_bn=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        # self.conv3_bn=nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256,1024, 3, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12800, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, embedding_dim) #adjust size according to the error here

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), (4,4))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (2,2))
        # x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), (2,2))
        # x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class pretrained_encoderCNN(nn.Module):
    def __init__(self, embedding_dim):
        super(pretrained_encoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        
        modules = list(resnet.children())[:-1] #change to -2 if pooling at the end is bad
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(2048, embedding_dim) #adjust size as necessary
        self.batch= nn.BatchNorm1d(embedding_dim,momentum = 0.01)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        out = features.reshape(features.size(0), -1)
        out = self.batch(self.embed(out)) if out.size(0) > 1 else self.embed(out)
        return out


class decoderRNN(nn.Module):
    def __init__(self, device, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(decoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size #hidden state
        self.device = device
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout = dropout if num_layers>1 else 0, batch_first=True) #lstm
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        h_0 = Variable(torch.zeros(self.num_layers, captions.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, captions.size(0), self.hidden_size)).to(self.device) #internal state
        x = self.embed(captions)
        x = self.dropout(torch.cat((features.unsqueeze(1), x), dim = 1))
        hiddens, _ = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = self.fc(hiddens)
        return out
    

class CaptionNet(nn.Module):
    def __init__(self, device, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pretrained, word_level):
        super(CaptionNet, self).__init__()
        self.device = device
        self.word_level = word_level
        
        
        self.encoder = pretrained_encoderCNN(embedding_dim) if pretrained else encoderCNN(embedding_dim)
        self.decoder = decoderRNN(device, vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        
    def forward(self, images, captions):
        x = self.encoder(images)
        x = self.decoder(x, captions)
        return x
    
    def caption(self, image, int_to_char, maxlen):
        result = ""
        caption = torch.Tensor([[1]]).type(torch.LongTensor).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(torch.Tensor(image).unsqueeze(0).to(self.device))
            
            for _ in range(maxlen):
                output = self.decoder(features, caption)
                pred = output.squeeze(0)[-1].argmax().unsqueeze(0)
                
                if int_to_char[pred.item()] == "<END>": #stop token
                    break
                    
                result += int_to_char[pred.item()] + (" " if self.word_level else "")
                caption = torch.cat((caption, pred.unsqueeze(0)), dim=1)
                
        return result
        
        
        
class captionGen:
    def __init__(self, img_size, big_data = False, word_level = False): #img_size should be 299 if using pretrained inception_v3 model
        self.history = {"train_loss":[], "validation_loss":[]}
        self.WORD_LEVEL = word_level
        
        if big_data:
            self.dataset = captionDataset(root="data/big_images/", annotations="data/big_captions.txt", img_size=img_size, word_level=word_level)
        else: 
            self.dataset = captionDataset(root="data/images/", annotations="data/captions.txt", img_size=img_size, word_level=word_level)
            
        self.train_data, self.test_data = torch.utils.data.random_split(self.dataset, [math.ceil(len(self.dataset)*.8),math.floor(len(self.dataset)*.2)])
        self.test_data, self.val_data = torch.utils.data.random_split(self.test_data, [math.ceil(len(self.test_data)*.5),math.floor(len(self.test_data)*.5)])
        print(f"{len(self.train_data)} training {len(self.test_data)} testing {len(self.val_data)} validation")
    
    def sample(self, count = 1):
        for _ in range(count):
            idx = random.randint(0,len(self.test_data)-1)
            plt.imshow(np.array(self.test_data[idx][0]*255).reshape(self.dataset.IMG_SIZE,self.dataset.IMG_SIZE,3).astype(np.uint8))
            plt.show()
            print(captioner.caption(self.test_data[idx][0]) + "\n")
    
    def define(self, embedding_dim, hidden_size, num_layers, dropout, pretrained):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.net = CaptionNet(self.device, len(self.dataset.vocab), embedding_dim, hidden_size, num_layers, dropout, pretrained, self.WORD_LEVEL).to(self.device)
        #print(f"\nUsing {self.device} device")
        if not pretrained:
            print(self.net)
        else:
            print(self.net.decoder)
    
    def train(self, batch_size, epochs, lr, verbose):
        self.BATCH_SIZE = batch_size
        dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        valdataloader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr = lr) #filter to avoid passing in the pretrained encoder
        loss_criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.char_to_int["<PAD>"])
        
        for epoch in range(epochs):
            loss_sum = 0
            for data in tqdm(dataloader, desc = "training") if verbose in [1,2] else dataloader:
                self.net.zero_grad()
                outputs = self.net(data[0].to(self.device), data[1].to(self.device)[:,:-1])
                loss = loss_criterion(outputs.reshape(-1,outputs.shape[2]), data[1].reshape(-1).to(self.device))
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            if verbose in [1,2]:
                print(f"loss:\t{loss_sum/len(dataloader)}")
            self.history["train_loss"] += [loss_sum/len(dataloader)]
            
            #clear gpu memory here
            
            val_loss_sum = 0
            for val_data in tqdm(valdataloader, desc = "validating") if verbose in [1,2] else valdataloader:
                with torch.no_grad():
                    outputs = self.net(val_data[0].to(self.device), val_data[1].to(self.device)[:,:-1])
                    val_loss_sum += loss_criterion(outputs.reshape(-1,outputs.shape[2]), val_data[1].reshape(-1).to(self.device)).item()
            if verbose in [1,2]:
                print(f"val loss:\t{val_loss_sum/len(valdataloader)}")
            self.history["validation_loss"] += [val_loss_sum/len(valdataloader)]
            
            if verbose == 2:
                self.sample()
    
    
    def curves(self):
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.history["train_loss"])
        plt.plot(self.history["validation_loss"])
        plt.show()
    
    def caption(self, image):
        return self.net.caption(image, self.dataset.int_to_char, self.dataset.maxlen)

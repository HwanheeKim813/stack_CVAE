import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import time
import pickle

from utils import time_since
from stackRNN import StackAugmentedRNN
from smiles_enumerator import SmilesEnumerator

import numpy as np
from random import *

## mini_Batch 
    
class stack_CVAE(nn.Module):
    def __init__(self, input_size, layer_type, hidden_size, latent_size, output_size,
                    max_sentence_length,  prop_size=0, batch_size=1, num_layers=1, bidirectional=False,lr = 0.001,
                    has_stack=False, stack_width=None, stack_depth=None,use_cuda=None):
    
        super(stack_CVAE, self).__init__()
        
        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prop_size = prop_size
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.lr = lr
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.data_encoder = nn.Embedding(input_size,hidden_size,padding_idx=0)  # embedding vector of 0 is zero vector
        rnn_input_size = self.hidden_size + self.prop_size
        decode_input_size = self.hidden_size + self.prop_size + latent_size
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
            print(self.use_cuda)   
        if self.bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        
        if layer_type == 'LSTM':
            self.encoder_rnn = StackAugmentedRNN(rnn_input_size, hidden_size, layer_type=self.layer_type,n_layers=num_layers,bidirectional=self.bidirectional,
                                                    has_stack=has_stack, stack_width=stack_width, stack_depth=stack_depth, use_cuda=self.use_cuda,)
            self.decoder_rnn = nn.LSTM(decode_input_size, hidden_size, n_layers=num_layers,bidirectional=self.bidirectional,batch_first=True)
            self.encoder_target_decoder = nn.Linear(hidden_size * self.num_dir, output_size)
            self.decoder_target_decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        elif layer_type == 'GRU':
            self.encoder_rnn = StackAugmentedRNN(rnn_input_size, hidden_size, layer_type=self.layer_type, n_layers=num_layers, is_bidirectional=self.bidirectional,
                                                    has_stack=has_stack, stack_width=stack_width, stack_depth=stack_depth, use_cuda=self.use_cuda)
            self.decoder_rnn = nn.GRU(decode_input_size, hidden_size, num_layers, bidirectional=self.bidirectional,batch_first=True)
            self.encoder_target_decoder = nn.Linear(hidden_size * self.num_dir, output_size)
            self.decoder_target_decoder = nn.Linear(hidden_size * self.num_dir, output_size)
            
        self.hidden2mean = nn.Linear(hidden_size * self.num_dir, latent_size)
        self.hidden2log = nn.Linear(hidden_size * self.num_dir, latent_size)
        
        self.latent2hidden = nn.Linear(latent_size,hidden_size * self.num_dir)
        self.output2vocab = nn.Linear(hidden_size*self.num_dir,output_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        
    def load_model(self,path):
        weights = torch.load(path)
        self.load_state_dict(weights)
        
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def change_lr(self,new_lr):
        self.lr = new_lr
        
    def init_hidden(self, batch_size):
        """
        Initialization of the hidden state of RNN.
        Returns
        -------
        hidden: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return torch.zeros((self.num_layers * self.num_dir, batch_size,
                                        self.hidden_size)).cuda()
        else:
            return torch.zeros((self.num_layers * self.num_dir, batch_size,
                                        self.hidden_size))								
    def init_cell(self):
        """
        Initialization of the cell state of LSTM. Only used when layers_type is
        'LSTM'
        Returns
        -------
        cell: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return torch.zeros((self.num_layers * self.num_dir, self.batch_size,
                                        self.hidden_size)).cuda()
        else:
            return torch.zeros((self.num_layers * self.num_dir, self.batch_size,
                                        self.hidden_size))
       
    def init_stack(self,batch_size):
        """
        Initialization of the stack state. Only used when has_stack is True
        Returns
        -------
        stack: torch.tensor
            tensor filled with zeros
        """
        result = torch.zeros((batch_size, self.stack_depth, self.stack_width))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
            
    def init_latent_vector(self, batch_size):
        latent_vector = torch.randn(batch_size, self.latent_size)
        if self.use_cuda:
            return latent_vector.cuda()
        else:
            return latent_vector
            

    def forward(self,inp,target,prop,encoder_hidden,decoder_hidden,encoder_stack,decoder_stack,batch_size=1,train=True):
        prop = prop.float()
        encoder_output = torch.tensor([]).cuda()
        embeded = self.data_encoder(inp)
        for c in range(inp.shape[1]):		 
            inputs = embeded[:,c]
            if self.prop_size:
                inputs = torch.cat((inputs,prop),dim=-1)
            output,encoder_hidden, encoder_stack = self.encoder_rnn(torch.unsqueeze(inputs,1),encoder_hidden, encoder_stack)
            encoder_output = torch.cat((encoder_output,output),dim=1)
        encoder_output_idx = self.encoder_target_decoder(encoder_output)
        
        mean = self.hidden2mean(encoder_hidden[-1])
        logv = self.hidden2log(encoder_hidden[-1])
        
        std = torch.exp(0.5 * logv)
        z = torch.randn_like(std)
        z = z * std + mean
        
        
        seq_length = embeded.shape[1]
        new_Z=torch.unsqueeze(z,1).repeat(1,seq_length,1)
        if self.prop_size:
            C = torch.unsqueeze(prop, 1)
            C = C.repeat(1,embeded.shape[1],1)
            decoder_input = torch.cat((new_Z,C,embeded),dim=-1)
        decoder_output, decoder_next_hidden = self.decoder_rnn(decoder_input,decoder_hidden)
        decoder_output_idx = self.decoder_target_decoder(decoder_output)
        

        KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        

        if train:
            decoder_rnn_loss = self.criterion(decoder_output_idx.view(-1,self.output_size),target.view(-1)) / batch_size
            KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
            total_loss = decoder_rnn_loss + KLD
            return total_loss, encoder_hidden, decoder_next_hidden
        else:
            return decoder_output_idx, encoder_hidden, decoder_next_hidden
        
    def encoder(self,inp,target,prop,encoder_hidden,encoder_stack,batch_size=1):
        prop = prop.view(1,-1).float()
        encoder_output = torch.tensor([]).cuda()
        embeded = self.data_encoder(inp.view(1,-1))
        for c in range(embeded.shape[1]):		 
            inputs = embeded[:,c]
            if self.prop_size:
                inputs = torch.cat((inputs,prop),dim=-1)
            output,encoder_hidden, encoder_stack = self.encoder_rnn(torch.unsqueeze(inputs,1),encoder_hidden, encoder_stack)
            encoder_output = torch.cat((encoder_output,output),dim=1)
        encoder_output_idx = self.encoder_target_decoder(encoder_output)
        
        mean = self.hidden2mean(encoder_hidden[-1])
        logv = self.hidden2log(encoder_hidden[-1])
        
        std = torch.exp(0.5 * logv)
        z = torch.randn_like(std)
        latent = z * std + mean
        KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        return latent, KLD
        
    def sample(self,latent_vector,inp,target,prop,decoder_hidden):
        embeded = self.data_encoder(inp.view(1, -1))
        z = latent_vector
        new_Z=torch.unsqueeze(z,1)
        if self.prop_size:
            C = prop.view(1,1,-1).float()
            decoder_input = torch.cat((new_Z,C,embeded),dim=-1)

        decoder_output, decoder_next_hidden = self.decoder_rnn(decoder_input,decoder_hidden)
        decoder_output_idx = self.decoder_target_decoder(decoder_output.view(1, -1))

        return decoder_output_idx, decoder_next_hidden
    
        
    def batch_sample(self,latent_vector,inp,target,prop,decoder_hidden):
        embeded = self.data_encoder(inp.view(-1,1))
        z = latent_vector
        
        seq_length = embeded.shape[1]
        new_Z=torch.unsqueeze(z,1).repeat(1,seq_length,1)
        if self.prop_size:
            C = torch.unsqueeze(prop, 1)
            C = C.repeat(1,embeded.shape[1],1)
            decoder_input = torch.cat((new_Z,C,embeded),dim=-1)
        decoder_output, decoder_next_hidden = self.decoder_rnn(decoder_input,decoder_hidden)
        decoder_output_idx = self.decoder_target_decoder(decoder_output)
        

        return decoder_output_idx, decoder_next_hidden
    
    def train_step(self,inp,target,prop):
        batch_size = inp.shape[0]
        if self.layer_type == 'LSTM':
            encoder_hidden = (self.init_hidden(batch_size),self.init_hidden(batch_size))
            decoder_hidden = (self.init_hidden(batch_size),self.init_hidden(batch_size))		  
        elif self.layer_type == 'GRU':
            encoder_hidden = self.init_hidden(batch_size)
            decoder_hidden = self.init_hidden(batch_size)
        if self.has_stack:
            encoder_stack = self.init_stack(self.batch_size)
            decoder_stack = self.init_stack(self.batch_size)
            
        else:
            stack = None
        self.optimizer.zero_grad()
        loss = 0
        vae_loss,encoder_hidden,decoder_hidden = self.forward(inp,target,prop,encoder_hidden,decoder_hidden,encoder_stack,decoder_stack,batch_size =batch_size)
        loss = vae_loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
        
        
    def sampleing(self, data, prop, prime_str='<', end_token='>', predict_len=100):  
        if self.layer_type == 'LSTM':
            encoder_hidden = (self.init_hidden(self.batch_size),self.init_hidden(self.batch_size))
            decoder_hidden = (self.init_hidden(self.batch_size),self.init_hidden(self.batch_size))
          
        elif self.layer_type == 'GRU':
            encoder_hidden = self.init_hidden(self.batch_size)
            decoder_hidden = self.init_hidden(self.batch_size)
        if self.has_stack:
            encoder_stack = self.init_stack(self.batch_size)
            decoder_stack = self.init_stack(self.batch_size)
        else:
            stack = None
        latent_vector = self.init_latent_vector(self.batch_size)
        self.optimizer.zero_grad()
        
        # prop = torch.tensor([180.04, 1.31, 63.6]).cuda()  
        if self.use_cuda:
            prop = torch.tensor([prop for _ in range(self.batch_size)]).cuda()
            prime_input = torch.tensor([[data.char_tensor(prime_str)] for _ in range(self.batch_size)]).cuda()
        else:
            prop = torch.tensor([prop for _ in range(self.batch_size)])
            prime_input = torch.tensor([[data.char_tensor(prime_str)] for _ in range(self.batch_size)])
        new_sample = [prime_str for _ in range(self.batch_size)]
        preds = []
        for p in range(len(prime_str)-1):
            _, encoder_hidden, decoder_hidden,encoder_stack = self.batch_sample(latent_vector,prime_input[p],[],prop,decoder_hidden)
        inp = prime_input.view(-1)
        for p in range(predict_len):
            output, decoder_hidden = self.batch_sample(latent_vector,inp,[],prop,decoder_hidden)
            probs = torch.softmax(output, dim=2)
            top_i = torch.multinomial(probs.view(self.batch_size,-1), 1)
            inp = torch.tensor([],dtype=torch.int64).cuda()
            top_i = top_i.view(-1).cpu().numpy()
            for n,i in zip(range(len(top_i)),top_i):
                predicted_char = data.all_characters[i]
                new_sample[n] += predicted_char
                inp = torch.cat((inp,data.char_tensor(predicted_char)),dim=0)
        return new_sample
        
        
    def evaluate(self, data, prop, prime_str='<', end_token='>', predict_len=100):
        if self.layer_type == 'LSTM':
            encoder_hidden = (self.init_hidden(1),self.init_hidden(1))
            decoder_hidden = (self.init_hidden(1),self.init_hidden(1))
          
        elif self.layer_type == 'GRU':
            encoder_hidden = self.init_hidden(1)
            decoder_hidden = self.init_hidden(1)
        if self.has_stack:
            encoder_stack = self.init_stack(1)
            decoder_stack = self.init_stack(1)
        else:
            stack = None
        latent_vector = self.init_latent_vector(1)
        
        self.optimizer.zero_grad()
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str
        if self.use_cuda:
            prop = torch.tensor(prop).cuda()
        else:
            prop = torch.tensor(prop)
        for p in range(len(prime_str)-1):
            _, encoder_hidden, decoder_hidden,encoder_stack = self.sample(latent_vector,prime_input[p],[],prop,decoder_hidden)
        inp = prime_input[-1]
        for p in range(predict_len):
            output, decoder_hidden = self.sample(latent_vector,inp,[],prop,decoder_hidden)
            probs = torch.softmax(output, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()
            predicted_char = data.all_characters[top_i]
            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if predicted_char == end_token:
                break
        return new_sample, latent_vector

    def get_random_batch(self,data,prop,batch_size):
        idx = np.random.randint(len(data.inputs), size = batch_size)
        inp = [data.inputs[i] for i in idx]
        tar = [data.targets[i] for i in idx]
        prop = [prop[i] for i in idx]
        for i in range(len(inp)):
            inp[i] = data.char_tensor(inp[i])
            tar[i] = data.char_tensor(tar[i])
            prop[i] = data.prop_tensor(prop[i])
        sorted_lengths = torch.LongTensor([torch.max(inp[i].data.nonzero()) + 1 for i in range(len(inp))])
        sorted_lengths, sorted_idx = sorted_lengths.sort(0, descending=True)
        inp = torch.stack(inp)
        tar = torch.stack(tar)
        prop = torch.stack(prop)
        inp = inp[sorted_idx]
        tar = tar[sorted_idx]
        prop = prop[sorted_idx]
        
        return inp,tar,prop,sorted_lengths
        
    
    def fit(self,data,n_iterations,eval_prop,start_epoch = 1,all_losses = [], print_every=3, plot_every=10,augment=False,save_dir='./model'):
        start = time.time()
        self.batch_iter_count = int(data.file_len / self.batch_size) - 1 
        iter_count = 150
        props = list(map(list,zip(*data.prop)))
        
        print("\t> number of iteraction per batch : %s" %iter_count)
        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None
        loss_avg = 0
        for epoch in range(start_epoch, n_iterations+1):
            losses = np.array([])
            for iter in range(iter_count):
                inputs, targets, prop,input_lengths = self.get_random_batch(data,props,self.batch_size)
                loss = self.train_step(inputs,targets,prop)
                loss_avg += loss
                
                if iter % print_every == 0:
                    print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,iter/iter_count*100, loss))
                    sample,latent = self.evaluate(data=data, prop=eval_prop,prime_str = '<', predict_len=100)
                    print(sample, '\n')
                if iter % plot_every == 0:
                    losses = np.append(losses,loss_avg / plot_every)
                    loss_avg = 0
            all_losses = np.append(all_losses,np.mean(losses))
            pth_path = save_dir+'/model_'+str(epoch)+'.pth'
            self.save_model(pth_path)
            with open(save_dir+'/all_losses_'+str(epoch)+'.pickle','wb') as f:
                pickle.dump(all_losses,f)
        return all_losses

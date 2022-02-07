import torch

import random
import numpy as np
from smiles_enumerator import SmilesEnumerator
from utils import read_smi_file, tokenize, read_object_property_file


class GeneratorData(object):
    def __init__(self, training_data_path, tokens=None, start_token='<', 
                 end_token='>', max_len=120, use_cuda=None,dif_len = False, **kwargs):
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        training_data_path: str
            path to file with training dataset. Training dataset must contain
            a column with training strings. The file also may contain other
            columns.

        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.

        max_len: int (default 120)
            maximum allowed length of the sequences. All sequences longer than
            max_len will be excluded from the training data.

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        kwargs: additional positional arguments
            These include cols_to_read (list, default [0]) specifying which
            column in the file with training data contains training sequences
            and delimiter (str, default ',') that will be used to separate
            columns if there are multiple of them in the file.

        """
        super(GeneratorData, self).__init__()

        if 'cols_to_read' not in kwargs:
            kwargs['cols_to_read'] = []
            
        self.col = 0
        datas = read_object_property_file(training_data_path,
                                                       **kwargs)
        if len(kwargs['cols_to_read'])>1:
            self.props=True 
            data = datas[:][0]
            self.prop = datas[:][1:]
        else:
            data = datas

        self.start_token = start_token
        self.end_token = end_token
        self.file = []
        self.inputs = []
        self.targets = []
        self.max_len = max_len
        ## data prepcoessing to remove smiles which have invalid token
        invalid_tokens = ['0', '%', '\n']
        print("[before removing invalid smiles] Number of SMILES: " + str(len(data)))
        data = [ele for ele in data if all(c not in invalid_tokens for c in ele)]
        print("[after removing invalid smiles] Number of SMILES: " + str(len(data)))
        
        
        for i in range(len(data)):
            if len(data[i]) <= max_len:
                self.file.append(self.start_token + data[i] + self.end_token)
                if dif_len:
                    self.inputs.append(self.start_token + data[i])
                    self.targets.append(data[i] + self.end_token)
                else:
                    self.inputs.append(self.start_token + data[i] + "D" * (self.max_len + 1 - len(data[i])))
                    self.targets.append(data[i] + self.end_token + "D" * (self.max_len + 1 - len(data[i])))
                    # self.inputs.append("E" * (self.max_len + 1 - len(data[i])) + self.start_token + data[i] )
                    # self.targets.append("E" * (self.max_len + 1 - len(data[i])) +data[i] + self.end_token)
        self.file_len = len(self.inputs)
        self.all_characters, self.char2idx,self.n_characters = tokenize(self.inputs+self.targets, tokens)
        self.use_cuda = use_cuda
        #for i in range(len(self.inputs)):
        #    if i % 100000 == 0: print(len(self.inputs),i)
        #    self.inputs[i] = self.char_tensor(self.inputs[i])
        #    self.targets[i] = self.char_tensor(self.targets[i])
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)

    def random_chunk(self):
        """
        Samples random SMILES string from generator training data set.
        Returns:
            random_smiles (str).
        """
        index = random.randint(0, self.file_len-1)
        return self.file[index]

    def char_tensor(self, string):
        """
        Converts SMILES into tensor of indices wrapped into torch.autograd.Variable.
        Args:
            string (str): input SMILES string
        Returns:
            tokenized_string (torch.autograd.Variable(torch.tensor))
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        if self.use_cuda:
            return torch.tensor(tensor).cuda()
        else:
            return torch.tensor(tensor)

    def random_training_set(self,smiles_augmentation):
        chunk = self.random_chunk()
        if smiles_augmentation is not None:
            chunk = '<' + smiles_augmentation.randomize_smiles(chunk[1:-1]) + '>'
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def read_sdf_file(self, path, fields_to_read):
        raise NotImplementedError
        
    def update_data(self, path):
        self.file, success = read_smi_file(path, unique=True)
        self.file_len = len(self.file)
        assert success
    
    def prop_tensor(self,prop):
        prop = np.array(prop,dtype=float)
        if self.use_cuda:
            return torch.tensor(prop).cuda()
        else:
            return torch.tensor(prop)
   
    def get_batch(self,iter,batch_size):
        inp = self.inputs[batch_size * iter : batch_size * (iter + 1) ]
        tar = self.targets[batch_size * iter : batch_size * (iter + 1) ]
        prop = list(map(list,zip(*self.prop)))
        prop = prop[batch_size * iter : batch_size * (iter + 1)]
        for i in range(len(inp)):
            inp[i] = self.char_tensor(inp[i])
            tar[i] = self.char_tensor(tar[i])
            prop[i] = self.prop_tensor(prop[i])
        sorted_lengths = torch.LongTensor([torch.max(inp[i].data.nonzero()) + 1 for i in range(len(inp))])
        sorted_lengths, sorted_idx = sorted_lengths.sort(0, descending=True)
        inp = torch.stack(inp)
        tar = torch.stack(tar)
        prop = torch.stack(prop)
        inp = inp[sorted_idx]
        tar = tar[sorted_idx]
        prop = prop[sorted_idx]
        
        return inp,tar,prop,sorted_lengths
        
    def get_batch_diflen(self,iter,batch_size):
        inp = self.inputs[batch_size * iter : batch_size * (iter + 1) ]
        tar = self.targets[batch_size * iter : batch_size * (iter + 1) ]
        batch_max_len = len(max(inp,key=len))
        for i in range(len(inp)):
            inp[i] = self.char_tensor(inp[i] + "D" * (batch_max_len + 1 - len(inp[i])))
            tar[i] = self.char_tensor(tar[i] + "D" * (batch_max_len + 1 - len(tar[i])))
        sorted_lengths = torch.LongTensor([torch.max(inp[i].data.nonzero()) + 1 for i in range(len(inp))])
        sorted_lengths, sorted_idx = sorted_lengths.sort(0, descending=True)
        inp = torch.stack(inp)
        tar = torch.stack(tar)
        inp = inp[sorted_idx]
        tar = tar[sorted_idx]
        return inp,tar,sorted_lengths
        


class PredictorData(object):
    def __init__(self, path, delimiter=',', cols=[0, 1], get_features=None,
                 has_label=True, labels_start=1, **kwargs):
        super(PredictorData, self).__init__()
        data = read_object_property_file(path, delimiter, cols_to_read=cols)
        if has_label:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = np.array(data[labels_start:], dtype='float32')
            self.y = self.y.reshape(-1, len(cols) - labels_start)
            if self.y.shape[1] == 1:
                self.y = self.y.reshape(-1)
        else:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = [None]*len(self.object)
        assert len(self.objects) == len(self.y)
        if get_features is not None:
            self.x, processed_indices, invalid_indices = \
                get_features(self.objects, **kwargs)
            self.invalid_objects = self.objects[invalid_indices]
            self.objects = self.objects[processed_indices]
            self.invalid_y = self.y[invalid_indices]
            self.y = self.y[processed_indices]
        else:
            self.x = self.objects
            self.invalid_objects = None
            self.invalid_y = None
        self.binary_y = None

    def binarize(self, threshold):
        self.binary_y = np.array(self.y >= threshold, dtype='int32')

import torch.nn as nn
import torch
from torch.autograd import Variable
#from gensim.models import KeyedVectors
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, embeds, hidden, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.encoder.weight.requires_grad = False
        #self.parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.rnn = nn.LSTM(ninp, nhid//2, nlayers, dropout=dropout, bidirectional=True)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights(tuple(embeds),ntoken,ninp)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        
    def init_weights(self,embeds,ntoken,ninp):
        initrange = 0.1

        k = len(embeds) # the first k indices are pretrained. the rest are unknown

        if k is not 0:
            first = np.array(embeds)
            second = np.random.uniform(-initrange,initrange,size=(ntoken-k,ninp))
            self.encoder.weight.data.copy_(torch.from_numpy(np.concatenate((first,second),axis=0)))
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = (self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*2, bsz, self.nhid//2).zero_()),
                Variable(weight.new(self.nlayers*2, bsz, self.nhid//2).zero_()))

class StackedRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, embeds, bottom_hidden, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(StackedRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.encoder.weight.requires_grad = False
        #self.parameters = filter(lambda p: p.requires_grad, self.parameters())
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        # self.encoder.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.encoder.weight.requires_grad = false
        # > parameters = filter(lambda p: p.requires_grad, net.parameters()) on optimizer to tell it to not grad encoder
        self.rnn_bottom = nn.LSTM(ninp, nhid//2, nlayers, dropout=dropout, bidirectional=True)
        self.mult_bottom = nn.Linear(nhid,nhid)
        self.tanh_bottom = nn.Tanh()
        self.rnn_top = nn.LSTM(ninp+nhid, nhid//2, nlayers, dropout=dropout, bidirectional=True)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.bottom_hidden = bottom_hidden

        self.init_weights(tuple(embeds),ntoken,ninp)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

        

        
    def init_weights(self,embeds,ntoken,ninp):
        initrange = 0.1

        k = len(embeds) # the first k indices are pretrained. the rest are unknown

        if k is not 0:
            first = np.array(embeds)
            second = np.random.uniform(-initrange,initrange,size=(ntoken-k,ninp))
            self.encoder.weight.data.copy_(torch.from_numpy(np.concatenate((first,second),axis=0)))
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.bottom_hidden is not None:
            #print("pls")
            self.rnn_bottom.weight_ih_l0 = self.bottom_hidden.rnn.weight_ih_l0
            self.rnn_bottom.weight_hh_l0 = self.bottom_hidden.rnn.weight_hh_l0
            self.rnn_bottom.bias_ih_l0 = self.bottom_hidden.rnn.bias_ih_l0
            self.rnn_bottom.bias_hh_l0 = self.bottom_hidden.rnn.bias_hh_l0
            self.rnn_bottom.weight_ih_l1 = self.bottom_hidden.rnn.weight_ih_l1
            self.rnn_bottom.weight_hh_l1 = self.bottom_hidden.rnn.weight_hh_l1
            self.rnn_bottom.bias_ih_l1 = self.bottom_hidden.rnn.bias_ih_l1
            self.rnn_bottom.bias_hh_l1 = self.bottom_hidden.rnn.bias_hh_l1
            #print("plsss")

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn_bottom(emb, hidden)
        output = self.drop(output)
        mult = self.mult_bottom(output.view(output.size(0)*output.size(1), output.size(2)))
        mult = self.tanh_bottom(mult)
        mult = mult.view(output.size(0), output.size(1), mult.size(1))

        weight = next(self.parameters()).data
        new_emb = Variable(weight.new(output.size(0), output.size(1), self.nhid+self.ninp).zero_())
        #concatenate each encoding by mult
        i = 0
        for em in emb:
            new_emb[i] = torch.cat((emb[i],mult[i]),dim=1)
            i+=1
        
        #reset hidden or no?
        output, hidden = self.rnn_top(new_emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*2, bsz, self.nhid//2).zero_()),
                Variable(weight.new(self.nlayers*2, bsz, self.nhid//2).zero_()))

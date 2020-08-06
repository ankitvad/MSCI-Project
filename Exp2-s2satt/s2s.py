#Exp1 +Attention on the set of ingredients.
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = True

class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, emb, use_emb = True):
		super().__init__()
		#Ingredients + Title
		self.embedding = nn.Embedding(input_dim, emb_dim)
		if use_emb:
			self.embedding.weight.data.copy_(emb)
			self.embedding.weight.requires_grad = True
		self.rnnI = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
		self.rnnT = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
		self.fc = nn.Linear(enc_hid_dim * 4, dec_hid_dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
	def forward(self, ttl, ing, ttl_len, ing_len):
		#src = [src len, batch size]
		#src_len = [src len]
		embeddedT = self.dropout(self.embedding(ttl))
		embeddedI = self.dropout(self.embedding(ing))
		#embedded = [src len, batch size, emb dim]
		packed_embeddedT = nn.utils.rnn.pack_padded_sequence(embeddedT, ttl_len, enforce_sorted = False)
		packed_embeddedI = nn.utils.rnn.pack_padded_sequence(embeddedI, ing_len, enforce_sorted = False)
		packed_outputsT, hiddenT = self.rnnT(packed_embeddedT)
		#Use Ingredients for Attention:
		packed_outputsI, hiddenI = self.rnnI(packed_embeddedI)
		#packed_outputs is a packed sequence containing all hidden states
		#hidden is now from the final non-padded element in the batch
		#outputsT, _ = nn.utils.rnn.pad_packed_sequence(packed_outputsT)
		outputsI, _ = nn.utils.rnn.pad_packed_sequence(packed_outputsI)
		#outputs is now a non-packed sequence, all hidden states obtained
		#when the input is a pad token are all zeros
		#outputs = [src len, batch size, hid dim * num directions]
		#hidden = [n layers * num directions, batch size, hid dim]
		#hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
		#outputs are always from the last layer
		#hidden [-2, :, : ] is the last of the forwards RNN
		#hidden [-1, :, : ] is the last of the backwards RNN
		#initial decoder hidden is final hidden state of the forwards and backwards
		#encoder RNNs fed through a linear layer
		hfT = torch.cat((hiddenT[-2,:,:], hiddenT[-1,:,:]), dim = 1)
		hfI = torch.cat((hiddenI[-2,:,:], hiddenI[-1,:,:]), dim = 1)
		hf = torch.cat((hfT, hfI), dim = 1)
		hidden = self.fc(hf)
		hidden = self.relu(hidden)
		#outputs = [src len, batch size, enc hid dim * 2]
		#hidden = [batch size, dec hid dim]
		return outputsI, hidden

class Attention(nn.Module):
	def __init__(self, enc_hid_dim, dec_hid_dim):
		super().__init__()
		self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
		self.v = nn.Linear(dec_hid_dim, 1, bias = False)
		self.relu = nn.ReLU()
	def forward(self, hidden, encoder_outputs, mask):
		#hidden = [batch size, dec hid dim]
		#encoder_outputs = [src len, batch size, enc hid dim * 2]
		batch_size = encoder_outputs.shape[1]
		src_len = encoder_outputs.shape[0]
		#repeat decoder hidden state src_len times
		hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
		encoder_outputs = encoder_outputs.permute(1, 0, 2)
		#hidden = [batch size, src len, dec hid dim]
		#encoder_outputs = [batch size, src len, enc hid dim * 2]
		energy = self.relu(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
		#energy = [batch size, src len, dec hid dim]
		attention = self.v(energy).squeeze(2)
		#attention = [batch size, src len]
		attention = attention.masked_fill(mask == 0, -1e10)
		return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
	def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout, attention, emb, use_emb = True):
		super().__init__()
		self.output_dim = output_dim
		self.attention = attention
		self.embedding = nn.Embedding(output_dim, emb_dim)
		if use_emb:
			self.embedding.weight.data.copy_(emb)
			self.embedding.weight.requires_grad = True
		#self.rnn = nn.GRU(dec_hid_dim + emb_dim, dec_hid_dim)
		self.rnn = nn.GRU((dec_hid_dim * 3) + emb_dim, dec_hid_dim)
		#self.fc_out = nn.Linear(dec_hid_dim, output_dim)
		self.fc_out = nn.Linear(dec_hid_dim * 3, output_dim)
		self.dropout = nn.Dropout(dropout)
	def forward(self, input, hidden, enc_hidden, encoder_outputs, mask):
			 
		#input = [batch size]
		#hidden = [batch size, dec hid dim]
		#encoder_outputs = [src len, batch size, enc hid dim * 2]
		#mask = [batch size, src len]
		input = input.unsqueeze(0)
		#input = [1, batch size]
		embedded = self.dropout(self.embedding(input))
		#embedded = [1, batch size, emb dim]
		#(self, hidden, encoder_outputs, mask)
		a = self.attention(hidden, encoder_outputs, mask)
		#a = [batch size, src len]
		a = a.unsqueeze(1)
		#a = [batch size, 1, src len]
		encoder_outputs = encoder_outputs.permute(1, 0, 2)
		#encoder_outputs = [batch size, src len, enc hid dim * 2]
		weighted = torch.bmm(a, encoder_outputs)
		#weighted = [batch size, 1, enc hid dim * 2]
		weighted = weighted.permute(1, 0, 2)
		#weighted = [1, batch size, enc hid dim * 2]
		rnn_input = torch.cat((embedded, weighted, enc_hidden.unsqueeze(0)), dim = 2)
		output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
		#output = [seq len, batch size, dec hid dim * n directions]
		#hidden = [n layers * n directions, batch size, dec hid dim]
		#seq len, n layers and n directions will always be 1 in this decoder, therefore:
		#output = [1, batch size, dec hid dim]
		#hidden = [1, batch size, dec hid dim]
		#this also means that output == hidden
		assert (output == hidden).all()
		embedded = embedded.squeeze(0)
		output = output.squeeze(0)
		weighted = weighted.squeeze(0)
		prediction = self.fc_out(torch.cat((output, weighted), dim = 1))
		#prediction = [batch size, output dim]
		return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, src_pad_idx, device):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_pad_idx = src_pad_idx
		self.device = device
	def create_mask(self, src):
		mask = (src != self.src_pad_idx).permute(1, 0)
		return mask
	def forward(self, ttl, ing, ttl_len, ing_len, trg, teacher_forcing_ratio = 0.5):
		#src = [src len, batch size]
		#src_len = [batch size]
		#trg = [trg len, batch size]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
		batch_size = ttl.shape[1]
		trg_len = trg.shape[0]
		trg_vocab_size = self.decoder.output_dim
		#tensor to store decoder outputs
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		#encoder_outputs is all hidden states of the input sequence, back and forwards
		#hidden is the final forward and backward hidden states, passed through a linear layer
		#ttl, ing, ttl_len, ing_len
		#(self, ttl, ing, ttl_len, ing_len)
		encoder_outputs, hidden = self.encoder(ttl, ing, ttl_len, ing_len)
		enc_hidden = hidden#first input to the decoder is the <sos> tokens
		input = trg[0,:]
		mask = self.create_mask(ing)
		#mask = [batch size, src len]
		for t in range(1, trg_len):
			#insert input token embedding, previous hidden state, all encoder hidden states
			#and mask
			#receive output tensor (predictions) and new hidden state
			#(self, input, hidden, enc_hidden, encoder_outputs, mask)
			output, hidden = self.decoder(input, hidden, enc_hidden, encoder_outputs, mask)
			#place predictions in a tensor holding predictions for each token
			outputs[t] = output
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			#get the highest predicted token from our predictions
			top1 = output.argmax(1)
			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[t] if teacher_force else top1
		return outputs
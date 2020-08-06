import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import random
import math
import time
from torchtext.data.metrics import bleu_score
from s2s import *
from torch.autograd import Variable
import random
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
#bleu_score: try nltk-smoothing function?

def translateData(sentence, src_field):
	tokens = [token.lower() for token in sentence]
	tokens = [src_field.init_token] + tokens + [src_field.eos_token]
	src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
	src_len = torch.LongTensor([len(src_indexes)]).to(device)
	return src_tensor, src_len

#translate_sentence(ttl, ing, TTL_ING, INS, model, device)
#ttl, ing, TTL_ING, INS, model, device, max_len
def translate_sentence(ttl, ing, src_field, trg_field, model, device, max_len = 150):
	model.eval()
	'''
	tokens = [token.lower() for token in sentence]
	tokens = [src_field.init_token] + tokens + [src_field.eos_token]
	src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
	src_len = torch.LongTensor([len(src_indexes)]).to(device)
	'''
	ttl_tensor, ttl_len = translateData(ttl, src_field)
	ing_tensor, ing_len = translateData(ing, src_field)
	with torch.no_grad():
		#hidden = self.encoder(ttl, ing, ttl_len, ing_len)
		encoder_outputs, hidden = model.encoder(ttl_tensor, ing_tensor, ttl_len, ing_len)
	mask = model.create_mask(ing_tensor)
	enc_hidden = hidden
	trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
	#attentions = torch.zeros(max_len, 1, ing_len.item()).to(device)
	for i in range(max_len):
		trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
		with torch.no_grad():
			#(self, input, hidden, enc_hidden, encoder_outputs, mask)
			output, hidden = model.decoder(trg_tensor, hidden, enc_hidden, encoder_outputs, mask)
		#attentions[i] = attention
		pred_token = output.argmax(1).item()
		trg_indexes.append(pred_token)
		if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
			break
	trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
	return trg_tokens[1:]#, attentions[:len(trg_tokens)-1]


def testOP(data, index, tp):
	x = open("eval.txt","a")
	x.write(tp+" Set: Index-"+str(index)+"\n")
	ttl = vars(data.examples[index])['title']
	ing = vars(data.examples[index])['ing']
	trg = vars(data.examples[index])['ins']
	translation = translate_sentence(ttl, ing, TTL_ING, INS, model, device)
	print(f'ttl = {ttl}')
	print(f'ing = {ing}')
	print(f'trg = {trg}')
	print(f'predicted trg = {translation}')
	x.write("ttl = "+" ".join(ttl).strip()+"\n")
	x.write("ing = "+" ".join(ing).strip()+"\n")
	x.write("Predicted trg = "+" ".join(translation).strip()+"\n")
	x.write("Ground trg = "+" ".join(trg).strip()+"\n")
	x.close()
	#display_attention(src, translation, attention, index, tp)

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 150):
	cc = SmoothingFunction()
	sentBleu = 0.0
	trgs = []
	pred_trgs = []
	for datum in tqdm(data):
		ttl = vars(datum)['title']
		ing = vars(datum)['ing']
		trg = vars(datum)['ins']
		#ttl, ing, TTL_ING, INS, model, device
		pred_trg = translate_sentence(ttl, ing, TTL_ING, INS, model, device, max_len)
		#cut off <eos> token
		pred_trg = pred_trg[:-1]
		sentBleu += sentence_bleu([trg], pred_trg, smoothing_function = cc.method4)
		pred_trgs.append(pred_trg)
		trgs.append([trg])
	sentBleu = sentBleu/len(data)
	corpusBleu = corpus_bleu(trgs, pred_trgs, smoothing_function = cc.method4)
	return sentBleu, corpusBleu

def prntBLEU(data, tp):
	sentBleu, corpusBleu = calculate_bleu(data, TTL_ING, INS, model, device)
	print(f'Sentence BLEU score = {sentBleu*100:.2f}')
	print(f'Corpus BLEU score = {corpusBleu*100:.2f}')
	x = open("eval.txt","a")
	x.write("\n\nSentence BLEU score for the "+tp+"-set is: "+str(f'{sentBleu*100:.2f}'))
	x.write("\n\nCorpus BLEU score for the "+tp+"-set is: "+str(f'{corpusBleu*100:.2f}'))
	x.close()


def getResults(model, test_iterator, criterion, tst, vld, trn, mName):
	model.load_state_dict(torch.load(mName))
	test_loss = evaluate(model, test_iterator, criterion)
	x = open("eval.txt","a")
	e = mName.split("epoch")[1].split(".")[0]
	x.write("\n\nEpoch Result: "+e+"\n")
	print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
	x.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
	x.close()
	#Random 10 samples.
	#example_idx = [10,20,50,100,200,1000,500,150,300,400]
	example_idx = len(test_iterator)
	example_idx = random.sample(range(example_idx), 20)
	#trn, tst, vld
	for i in example_idx:
		testOP(tst, i, "Test")
	#Printing BLEU score:
	prntBLEU(tst, "Test")
	#prntBLEU(vld, "Validation")
	#prntBLEU(trn, "Train")



def init_weights(m):
	for name, param in m.named_parameters():
		if 'weight' in name:
			nn.init.normal_(param.data, mean=0, std=0.01)
		else:
			nn.init.constant_(param.data, 0)


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0
	batchStartTime = time.time()
	for i, batch in enumerate(iterator):
		#title,ing,ins
		if i%100 == 0:
			batchEndTime = time.time()
			epoch_mins, epoch_secs = epoch_time(batchStartTime, batchEndTime)
			batchLoss = float(epoch_loss/(i+1))
			print(f'Batch: {i+1:02} | Time: {epoch_mins}m {epoch_secs}s | Loss: {batchLoss:.4f}')
			batchStartTime = time.time()
			#print("Batch: "+str(i)+", Loss: "+str(float(epoch_loss/(i+1)))+"\n")
		ttl, ttl_len = batch.title
		ing, ing_len = batch.ing
		trg = batch.ins
		optimizer.zero_grad()
		#(self, ttl, ing, ttl_len, ing_len, trg, teacher_forcing_ratio = 0.5)
		output = model(ttl, ing, ttl_len, ing_len, trg)
		#trg = [trg len, batch size]
		#output = [trg len, batch size, output dim]
		output_dim = output.shape[-1]
		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)
		#trg = [(trg len - 1) * batch size]
		#output = [(trg len - 1) * batch size, output dim]
		loss = criterion(output, trg)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		epoch_loss += loss.item()
	return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
	model.eval()
	epoch_loss = 0
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			ttl, ttl_len = batch.title
			ing, ing_len = batch.ing
			trg = batch.ins
			output = model(ttl, ing, ttl_len, ing_len, trg, 0.0)
			#output = model(src, src_len, trg, 0) #turn off teacher forcing
			#trg = [trg len, batch size]
			#output = [trg len, batch size, output dim]
			output_dim = output.shape[-1]
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			#trg = [(trg len - 1) * batch size]
			#output = [(trg len - 1) * batch size, output dim]
			loss = criterion(output, trg)
			epoch_loss += loss.item()
	return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

if __name__ == '__main__':
	mType = sys.argv[1]#train/test
	SEED = 1234
	BATCH_SIZE = 32
	ENC_EMB_DIM = 200
	DEC_EMB_DIM = 200
	ENC_HID_DIM = 256
	DEC_HID_DIM = 256
	ENC_DROPOUT = 0.3
	DEC_DROPOUT = 0.3
	N_EPOCHS = 21
	CLIP = 1
	LR = 3e-4
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	tokenize = lambda x: x.split()
	TTL_ING = Field(tokenize = "spacy", tokenizer_language="en", init_token='<sos>', eos_token='<eos>', lower = True, include_lengths = True)
	INS = Field(tokenize = "spacy", tokenizer_language="en", init_token='<sos>', eos_token='<eos>', lower = True, include_lengths = False)
	#title,ing,ins
	tv_datafields = [("title", TTL_ING), ("ing", TTL_ING), ("ins", INS)]
	trn, vld, tst = TabularDataset.splits(path = "../data", train = 'train.tsv', validation="valid.tsv", test = "test.tsv", format='tsv', skip_header = True, fields = tv_datafields)
	TTL_ING.build_vocab(trn, vld, tst, min_freq=2, max_size=50000, vectors="glove.6B.200d")
	INS.build_vocab(trn, vld, tst, min_freq=2, max_size=50000, vectors="glove.6B.200d")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_iterator, valid_iterator, test_iterator = BucketIterator.splits((trn, vld, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.ing), device = device)
	INPUT_DIM = len(TTL_ING.vocab)
	OUTPUT_DIM = len(INS.vocab)
	SRC_PAD_IDX = TTL_ING.vocab.stoi[TTL_ING.pad_token]
	attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
	#(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, emb, use_emb = True)
	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, TTL_ING.vocab.vectors, use_emb = True)
	#(self, output_dim, emb_dim, dec_hid_dim, dropout, attention, emb, use_emb = True)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, INS.vocab.vectors, use_emb = True)
	#(self, encoder, decoder, src_pad_idx, device)
	model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
	best_valid_loss = float('inf')
	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), lr = LR)
	TRG_PAD_IDX = INS.vocab.stoi[INS.pad_token]
	criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
	x = open("result.txt","a")
	print(f'The model has {count_parameters(model):,} trainable parameters')
	x.write(f'The model has {count_parameters(model):,} trainable parameters\n')
	x.close()
	if mType == "train":
		for epoch in range(N_EPOCHS):
			x = open("result.txt","a")
			start_time = time.time()
			train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
			valid_loss = evaluate(model, valid_iterator, criterion)
			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			elif (epoch % 5 == 0):
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			x.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
			print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
			x.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
			print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
			x.write(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
			x.close()
		#Printing/Calculating Results:
	elif mType == "test":
		getResults(model, test_iterator, criterion, tst, vld, trn, mName = 'tut-model-epoch'+str(sys.argv[2])+'.pt')
	elif mType == "continue":
		nE = sys.argv[2]
		mName = 'tut-model-epoch'+str(nE)+'.pt'
		model.load_state_dict(torch.load(mName))
		for epoch in range(int(nE)+1, N_EPOCHS):
			x = open("result.txt","a")
			start_time = time.time()
			train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
			valid_loss = evaluate(model, valid_iterator, criterion)
			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			elif (epoch % 5 == 0):
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			x.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
			print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
			x.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
			print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
			x.write(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
			x.close()
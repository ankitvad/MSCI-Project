import spacy
import pickle as pl

nlp = spacy.load("en_core_web_sm")

fl = ['procData/recipe1m_test.pkl', 'procData/recipe1m_val.pkl', 'procData/recipe1m_train.pkl']

verbs = {}

for f in fl:
	t = pl.load(open(f,"rb"))
	for i in t:
		ins = i['instructions']
		ID = i['id']
		verbs[ID] = []
		for j in ins:
			doc = nlp(j)
			tmp = [tok.text for tok in doc if tok.pos_ == "VERB"]
			verbs[ID].append(tmp)

with open('procData/id_verb.pkl', 'wb') as handle:
	pl.dump(verbs, handle, protocol = pl.HIGHEST_PROTOCOL)


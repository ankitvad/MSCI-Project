import spacy

nlp = spacy.load("en_core_web_sm")

for i in t:
	doc = nlp(i)
	for token in doc:
		print (token.text, token.tag_, token.head.text, token.dep_)
	print("\n")

# x = the whole thing.

ins = t['instructions']
ing = t['ingredients']
title = t['title']

tmp = {}
for k in x.keys():
	ins = x[k]['instructions']
	ing = x[k]['ingredients']
	title = x[k]['title']
	for j in ing:
		doc = nlp(j)
		tok = [t for t in doc if "NN" in t.tag_]
		for t in tok:
			tmp[t.text.lower()] = []
			c = [i for i in t.children]
			if c:
				for i in c:
					if "JJ" in i.tag_:
						tmp[t].append(i)


ds = []

for i in DS:
	ing = i["ingredients"]
	title = i["title"]
	ins = i["instructions"]
	ing = " ".join(ing).rstrip().lstrip()
	title = " ".join(title).rstrip().lstrip()
	ins = " ".join(ins).rstrip().lstrip()
	ds.append([title,ing,ins])

def wOut(a, nm):
	x = open(nm+".tsv", "w")
	x.write("Title\tIngredients\tInstructions\n")
	for i in a:
		x.write(i[0]+"\t"+i[1]+"\t"+i[2]+"\n")
	x.close()

import spacy
import glob
nlp = spacy.load("en_core_web_sm")

fl = glob.glob("*.tsv")

d = {}
c = 1

for f in fl:
	x = open(f, "r").read().split("\n")[1:]
	partition = f.replace(".tsv","")
	for i in x:
		if i:
			title, ing, ins = i.split("\t")
			ingCheck = {nm:"" for nm in ing.replace("_"," ").split()}
			insSent = ins.split(".")
			sentPart = []
			for j in insSent:
				NN = []
				VB = []
				doc = nlp(j+".")
				for tok in doc:
					if tok.pos_ == "VERB":
						VB.append(tok.text)
					if tok.pos_ == "NOUN":
						if tok.text in ingCheck:
							NN.append(tok.text)
				sentPart.append([doc.text, NN, VB])
			d[c] = {
				"partition": partition,
				"title": title,
				"ing": ing,
				"ins": ins,
				"sentPart": sentPart
			}
			c+=1

with open('noun_verb.pkl', 'wb') as handle:
	pl.dump(d, handle, protocol = pl.HIGHEST_PROTOCOL)

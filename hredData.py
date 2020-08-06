import glob

fl = glob.glob("*.tsv")
header = "Title\tIngredients\tSource\tTarget\tLast"

def cleanQuote(s):
	#return s.lstrip('"').lstrip("'").rstrip('"').rstrip("'").lstrip().rstrip()+"."
	return s.replace('"', "").lstrip().rstrip()+"."

for f in fl:
	nm = f.split(".")[0]+"_separated.tsv"
	y = open(nm, "w")
	y.write(header+"\n")
	x = open(f, "r").read().split("\n")[1:-1]
	for i in x:
		i = i.split("\t")
		INS = i[-1].split(".")
		if len(INS) > 2:
			INS = [cleanQuote(i) for i in INS]
			INS[-1] += " <eou>"
			insPair = []
			for s,t in zip(INS, INS[1:]):
				insPair.append([i[0], i[1], s, t, str(0)])
			insPair[-1][-1] = str(1)
			for j in insPair:
				y.write("\t".join(j)+"\n")
	y.close()
import pandas

df = pandas.read_csv("input/2010-02-05_60seconds.csv")

for i in df:
	print df[i]

import os

dir_name = "./n2c2 Data/data/annotations/"
test = os.listdir(dir_name)

for filename in test:
	noExtension = filename.split(".")[0]
	if filename[0] != ".":
		os.rename(dir_name+filename, dir_name+noExtension)
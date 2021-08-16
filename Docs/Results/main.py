
def delblankline(infile, outfile):
	infopen = open(infile, 'r', encoding="utf-8")
	outfopen = open(outfile, 'w', encoding="utf-8")

	lines = infopen.readlines()
	for line in lines:
		if line.split():
			outfopen.writelines(line)
		else:
			outfopen.writelines("")

	infopen.close()
	outfopen.close()


if "__name__" == "__main__":
	delblankline("./raw/1_epoch/original_1_epoch_Evaluation.txt", "./clean/1_epoch/original_1_epoch_Evaluation.txt")

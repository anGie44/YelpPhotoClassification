import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def create_label_collection(file):
	df = pd.read_csv(file, delimiter=',', header=0)
	y = df.iloc[:, 1].values

	for i in range(len(y)):
		try:
			y[i] = np.fromstring(y[i], dtype=int, sep=' ')
		except:
			y[i] = []
	MultiLabelBinarizer().fit_transform(y)

def main():
	create_label_collection(sys.argv[1])

if __name__ == "__main__":
	main()




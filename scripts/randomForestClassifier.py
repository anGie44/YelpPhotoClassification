import numpy as np
import pandas as pd
import cv2
from PIL import Image
import cv2
from subprocess import check_output
#from sklearn.neural_network import MLPClassifier  only available in dev version
from sklearn.ensemble import RandomForestClassifier
import pprint
from numpy import genfromtxt, savetxt
import csv
import time
import config
import xgboost as xgb

#function to return a 1D numpy array
def img_as_array(image_id, test_indicator, new_img_size):
	if not test_indicator:
		imagepath = '../input/train_photos/' + str(image_id) + '.jpg'
	else:
		imagepath = '../input/test_photos/' + str(image_id) + '.jpg'
	img = cv2.imread(imagepath)
	resized_image = cv2.resize(img, new_img_size)

	resized_image = resized_image.reshape(3 * new_img_size[0] * new_img_size[1])
	more_features = other_features(resized_image, new_img_size)
	avg = averages(resized_image, new_img_size)
	tmp = np.append(resized_image, more_features)

	return np.append(tmp, avg)

def averages(arr, img_size):
	img = np.reshape(arr, (img_size[0], img_size[1], 3))
	uAvg = np.average(img, axis=0).tolist()
	vAvg = np.average(img, axis=1).tolist()
	wAvg = np.average(img, axis=2).tolist()

	out = []
	for x in uAvg+vAvg+wAvg:
		out += x
	return np.array(out)

def other_features(arr, img_size):
	img = np.reshape(arr, (img_size[0], img_size[1], 3))
	u = np.sum(img, axis=0).tolist()
	v = np.sum(img, axis=1).tolist()
	w = np.sum(img, axis=2).tolist()

	out = []
	for x in u+v+w:
		out += x

	return np.array(out)

def data_collection_stats():
	print(check_output(["ls", "../input"]).decode("utf8"))
	train_images = check_output(["ls", "../input/train_photos"]).decode("utf8")
	print(train_images[:])
	print('time elapsed ' + str((time.time() - config.start_time)/60))

	print('Reading data...')
	train_photos = pd.read_csv('../input/train_photo_to_biz_ids.csv')
	train_photos.sort_values(['business_id'], inplace=True)
	train_photos.set_index(['business_id'])

	test_photos = pd.read_csv('../input/test_photo_to_biz.csv')
	test_photos.sort_values(['business_id'], inplace=True)
	test_photos.set_index(['business_id'])

	train = pd.read_csv('../input/train.csv')
	train.sort_values(['business_id'], inplace=True)
	train.reset_index(drop=True)

	print('Number of training samples: ', train.shape[0])
	print('Number of test samples: ', len(set(test_photos['business_id'])))
	print('Finished reading data...')
	print('Time elapsed: ' + str((time.time() - config.start_time)/60))

	print('Reading/Modifying images..')

	return (train_photos, test_photos, train)

#column_names = ['A_'+str(i) for i in range(arr_size)] + ['B_'+str(i) for i in range(arr_size)] +['C_'+str(i) for i in range(arr_size)]

def training_data_prep(train_photos, train):
	print('\tPreparing train data...')

	for row in train_photos.itertuples():
		image_id = row[1]
		loc = row[2]
		print(row)
		print(loc, config.pLoc)
		if loc == config.pLoc:
			config.count += 1
			if config.count < config.imgs_per_loc:
				config.arr += list(img_as_array(image_id, False, config.img_size))
				#print(row(1))
				config.i += 1
			else:
				continue
		else:
			if config.arr:
				config.locs.append(loc)
				config.X.append([int(x) for x in config.arr])
				#print(loc, type(loc))
				y_vals = train[train['business_id'] == loc]
				y = [0]*9
				for r in y_vals.itertuples():
					try:
						for u in [int(x) for x in r[2].split(' ')]:
							y[u] = 1
					except:
						print(r)
				config.Y.append(y)
				#print(len(config.X), len(config.Y))
			#print(len(config.arr))
			#print(config.arr)
			config.pLoc = loc
			config.arr = list(img_as_array(image_id, False, config.img_size))
			config.count = 1
			config.i += 1
		if config.i > config.max_images:
			break

def testing_data_prep(test_photos):
	print('\tPreparing test data...')


	for row in test_photos.itertuples():
		image_id = row[1]
		loc = row[2]
		print(row)
		print(loc, config.pLoc)
		if loc == config.pLoc:
			config.count += 1
			if config.count < config.imgs_per_loc:
				config.arr += list(img_as_array(image_id, True, config.img_size))
				#print(row(1))
				config.i += 1
			else:
				continue
		else:
			if config.arr:
				config.locs.append(loc)
				config.X_test.append([int(x) for x in config.arr])
				config.test_ids.append(loc)
			config.pLoc = loc
			config.arr = list(img_as_array(image_id, True, config.img_size))
			config.count = 1
			config.i += 1
		if config.i > config.max_images:
			continue

def train_and_predict():
	print('Converting data...')

	config.X = np.array(config.X)
	config.Y = np.array(config.Y)
	config.X_test = np.array(config.X_test)
	#print(config.X.shape)
	#print(config.Y.shape)
	#print(config.X_test.shape)
	print('Training...')
	print('Time Elapsed: ' + str((time.time() - config.start_time)/60))

	num_classes = len(config.Y[1, :])

	for i in range(num_classes):
		print('Creating Classifier: ', i)
		rf = RandomForestClassifier(n_estimators=300, max_depth=2*config.img_size[0], n_jobs=-1, oob_score=True, verbose=2, criterion="entropy")
		gbm = xgb.XGBClassifier(max_depth=2*config.img_size[0], n_estimators=300, learning_rate=0.05)

		print('Fitting Random Forest Classifier: ', i)
		rf.fit(config.X, config.Y[:, i])

		print('Fitting With XGBoost Classifier: ', i)
		gbm.fit(config.X, config.Y[:, i])

		print('Getting Random Forest Predictions for attribute: ', i)
		y_pred_rf = rf.predict(config.X_test)
		config.Y_pred_rf.append(y_pred_rf)

		print('Getting XGBoost Predictions for attribute: ', i)
		y_pred_xgb = gbm.predict(config.X_test)
		config.Y_pred_xgb.append(y_pred_xgb)

		print(y_pred_rf)
		print(y_pred_xgb)


def prepare_output():
	print('Preparing Output...')

	config.Y_pred_rf = np.vstack(config.Y_pred_rf)
	print(config.Y_pred_rf.shape)
	config.Y_pred_rf = np.transpose(config.Y_pred_rf)
	print(config.Y_pred_rf.shape)
	config.Y_pred_rf = config.Y_pred_rf.tolist()

	config.Y_pred_xgb = np.vstack(config.Y_pred_xgb)
	print(config.Y_pred_xgb.shape)
	config.Y_pred_xgb = np.transpose(config.Y_pred_xgb)
	print(config.Y_pred_xgb.shape)
	config.Y_pred_xgb = config.Y_pred_xgb.tolist()


	with open("training_labels.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['business_id', 'labels'])
		for i, r in enumerate(zip(config.test_ids, config.Y)):
			output = ' '.join([str(j) if x > 0 else '' for j,x in enumerate(r[1])]).strip()
			line = [r[0], output]
			print(line)
			print(r)
			writer.writerow(line)

	with open("rfpredictions_" + config.tag + ".csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['business_id', 'labels'])
		for i, r in enumerate(zip(config.test_ids, config.Y_pred_rf)):
			output = ' '.join([str(j) if x > 0 else '' for j,x in enumerate(r[1])]).strip()
			line = [r[0], output]
			print(line)
			print(r)
			writer.writerow(line)


	with open("xgbpredictions_" + config.tag + ".csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['business_id', 'labels'])
		for i, r in enumerate(zip(config.test_ids, config.Y_pred_xgb)):
			output = ' '.join([str(j) if x > 0 else '' for j,x in enumerate(r[1])]).strip()
			line = [r[0], output]
			print(line)
			print(r)
			writer.writerow(line)

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
	acc_list = []
	for i in range(y_true.shape[0]):
		set_true = set(np.where(y_true[i])[0])
		set_pred = set(np.where(y_pred[i])[0])
		tmp_a = None
		if len(set_true) == 0 and len(set_pred) == 0:
			tmp_a = 1
		else:
			tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
		acc_list.append(tmp_a)
	return np.mean(acc_list)

def main():
	train_fotos, test_fotos, train_d = data_collection_stats()
	training_data_prep(train_fotos, train_d)
	testing_data_prep(test_fotos)
	train_and_predict() #trains 2 classifiers: random forest, forest with xgboost
	prepare_output()
	print('Accuracy Score w/Random Forest Classifier: ' % hamming_score(config.Y, config.Y_pred_rf))
	print('Accuracy Score w/XGBoost Classifier: ' % hamming_score(config.Y, config.Y_pred_xgb))


#if __name__ == "__main__":
#	main()

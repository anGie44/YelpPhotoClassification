import cv2
import os

class YelpImageDataset(object):
	def __init__(self, path):
		self.train_imgs = []
		self.feature_mat = []
		images = os.listdir(path)	
		for i in images:
			img = cv2.imread(i, 0)
			self.train_imgs.append(img)

	def _define_surf_features:
		surf = cv2.xfeatures2d.SURF_create(400)
		surf.setExtended(True)
		for i in range(len(self.train_imgs)):
			(kp, des) = surf.detectAndCompute(self.train_imgs[i], None)
			self.feature_mat.append((kp,des))

	

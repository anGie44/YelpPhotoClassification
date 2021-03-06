import os
import cv2

class YelpImageDataset(object):
	def __init__(self, path):
		self.train_imgs = []
		self.surf_features = []
		self.sift_features = []
		images = os.listdir(path)	
		for i in range(len(images)):
			newpath = '/%s' % images[i]
			newpath = path + newpath
			img = cv2.imread('%s' % newpath, 0)
			self.train_imgs.append(img)

	def define_surf_features(self):
		surf = cv2.xfeatures2d.SURF_create(400)
		surf.setExtended(True)
		for i in range(len(self.train_imgs)):
			(kp, des) = surf.detectAndCompute(self.train_imgs[i], None)
			self.surf_features.append((kp,des))

	def define_sift_features(self):
		sift = cv2.xfeatures2d.SIFT_create()
		for i in range(len(self.train_imgs)):
			kp = sift.detectAndCompute(self.train_imgs[i], None)
			self.sift_features.append(kp)


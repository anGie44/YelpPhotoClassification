import time

start_time = round(time.time())
tag = str(start_time)
img_size = (20,20)
arr_size = 3 * img_size[0]*img_size[1]
i = 0
max_images = 1200
imgs_per_loc = 1 # use only 1 image for each restaurant
pLoc = -1       
count = 0       
X = []          
X_test = []     
arr = []                
Y = []                  
Y_pred_rf = []                     
Y_pred_xgb = []                     
locs = []                       
test_ids = []


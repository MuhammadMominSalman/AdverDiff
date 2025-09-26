import argparse
import os
import skimage
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')

opt = parser.parse_args()

# directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
dist_list = []
for file in files:
	file2 = file.split('_')[-1]
	if(os.path.exists(os.path.join(opt.dir1,file2))):
		# Load images
		img0 = np.array(Image.open(os.path.join(opt.dir0,file)).convert("RGB")) # RGB image from [-1,1]
		img1 = np.array(Image.open(os.path.join(opt.dir1,file2)).convert("RGB"))
		
		# Compute distance
		dist01 = skimage.metrics.structural_similarity(img0,img1,channel_axis=2)

		dist_list.append(dist01)
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
f.writelines('Avg Value: %.6f\n'%(sum(dist_list)/len(dist_list)))
f.close()

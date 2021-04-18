import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import pickle
from pypianoroll import Multitrack, Track
from numpy.random import random
import time

def generate(result):
	
	newResult = concatenate(result)
	convert2mid(newResult)

def read_result(result_path):

	result = []
	file = open(result_path, 'rb')
	result = pickle.load(file)
	file.close()
	result = np.array(result)
	return result

def concatenate(result):

	newResult = []
	for image in result:
		newImg = []
		for piece in image:
			newPiece = []
			tmp1 = np.zeros((1000, 24), dtype=int)
			tmp2 = np.concatenate((tmp1, piece[:,:,-1]), axis=1)
			tmp3 = np.zeros((1000, 20), dtype=int)
			tmp4 = np.concatenate((tmp2, tmp3), axis=1)
			newPiece.append(tmp4)
			newImg.append(newPiece)

		newResult.append(newImg)

	newResult = np.array(newResult)
	return newResult

def convert2mid(newResult):

	for k in range(8):
		final_result = np.array(newResult[-1][k])
		final_arr = final_result[0,:,:]
		new_arr = [[0]*128]*1000
		new_arr = np.array(new_arr)
		for i in range(1000):
			for j in range(128):
				if final_arr[i][j] <= 0.98:
					new_arr[i][j] = 0
				else:
					new_arr[i][j] = 100

		# Plot the results for verification

		# fig = plt.figure()
		# fig.set_size_inches(18,10)
		# plt.imshow(new_arr.T, aspect='auto', cmap='gray')
		# plt.gca().invert_yaxis()
		# plt.show()
		
		finalTrack = Track(pianoroll=new_arr, program=0, is_drum=False, name='piano')
		fig, ax = finalTrack.plot(cmap='Blues')
		multi = Multitrack(tracks=[finalTrack])
		multi.write('./result_demo_'+str(k)+'.mid')
	print("midi files conversion completed!")


# main program driver

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate GAN music.')
	parser.add_argument('string', metavar='result_path', type=str, nargs='?',
					help='the path name of the training result')
	args = parser.parse_args()
	
	try:
		result_path = sys.argv[1:][0]
		result = read_result(result_path)
		print(result.shape)
		generate(result)
	except:
		print("Please specify the \"training result\" as an argument.")
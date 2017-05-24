#reading a dataset

import cv2
import csv
import math
import numpy as np

"""
def read_jaffe_dataset(path):
	with open("data/japaneseWomen_data.csv") as file_obj:
		reader = csv.reader(file_obj, delimiter=',')
		flag = 0 

		if reader is None:
			print("Could not find the data file")
		img_data = [] 
		img_labels = []
		for line in reader:
			if str(line[0][0:10]) == "KL.AN2.168":
				flag = 1
			if str(line[0][0:9]) == "KM.AN1.17":
				flag = 0

			if flag == 0: 
			   img_data.append(cv2.imread(path+str(line[0][0:len(line[0])-1]),0))
			#   print(path+str(line[0][0:len(line[0])-1])) ;
			   if line[0] is None:
					print("No data found in the csv file")
			   if img_data[-1] is None:
					print("Image cannot be read ::",str(line[0][0:len(line[0])-1]))
			elif flag == 1:
			   img_data.append(cv2.imread(path+str(line[0][0:len(line[0])]),0))
			   print(path+str(line[0][0:len(line[0])])) ;
			   if line[0] is None:
					print("No data found in the csv file")
			   if img_data[-1] is None:
					print("Image cannot be read ::",str(line[0][0:len(line[0])]))

			print(line[1])
			img_labels.append(line[1])
			cv2.imshow("Image",img_data[-1]) 
			cv2.waitKey(2)
			  
	cv2.destroyAllWindows()
"""

def extract_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces,numFaces =face_cascade.detectMultiScale2(img,1.3,5);

    for x,y,w,h in faces :
       # cv2.rectangle(img,(x+18,y+30),(x+w-28,y+h-10),(255,0,0),2) ;
        face_img = img[y+30:y+h-10,x+18:x+w-28]
        face_img = cv2.resize(face_img,(100,100),interpolation = cv2.INTER_CUBIC)
    return face_img


def read_jaffe_dataset(path):
	with open("data/japaneseWomen_data.csv") as file_obj:
		reader = csv.reader(file_obj, delimiter=',')
		image_name = [] 
		image_label = [] 
		for line in reader:
			image_name.append(line[0])
			image_label.append(line[1])

		print("Image lengt = ", len(image_name),len(image_label))

	from os import listdir
	from os.path import isfile, join 
	img_data = [] 
	img_label = []
	onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
	for j in onlyfiles:
		indices = [i for i, s in enumerate(image_name) if j in s]
		
		if len(indices) == 0:
			print("Multiple or no indices found :: ", j ,indices);
			continue
		else:
			#print("j = ", j )
			#print(indices)
			temp_img_1 = cv2.imread(path+str(j),0)
			temp_img = extract_face(temp_img_1)
			if temp_img is  None:
				continue
			img_data.append(temp_img)
			#print("image size = ",len(img_data))

			cv2.imshow("img",img_data[-1])
			cv2.waitKey()
			img_label.append(image_label[indices[0]])
	sorted_data = []; 
	sorted_label = [] ;
	for i in range(1,8):
		temp_data = [];
		temp_label = [];
		for m in range(0,len(img_label)):
			#print(img_label[m],i)
			if img_label[m] == str(i) :
				temp_label.append(img_label[m]);
				temp_data.append(img_data[m]);
				#print("found match")
		sorted_data.append(temp_data)
		sorted_label.append(temp_label)

	#for i in range(0,7):
	#	print("Length of data = ",len(sorted_data[i]),len(sorted_label[i]))
	jaffe_trainData = [] 
	jaffe_trainLabels = [] 
	jaffe_testData = [] 
	jaffe_testLabels=[]
	print(int(math.ceil(len(sorted_label[0])*0.7)) , len(sorted_label[0]))
	for i in range(0,7):
		for j in range(0,int(math.ceil(len(sorted_label[i])*0.7))):
			jaffe_trainData.append(sorted_data[i][j])
			jaffe_trainLabels.append(sorted_label[i][j])

		for j in range(int(math.ceil(len(sorted_label[i])*0.7)),len(sorted_label[i])):
			jaffe_testData.append(sorted_data[i][j])
			jaffe_testLabels.append(sorted_label[i][j])

	print("Jaffe = ",len(jaffe_trainData),len(jaffe_trainLabels),len(jaffe_testLabels),len(jaffe_testData))


	#print(sorted_data[1][2],sorted_label[1])
	np.save('data/jaffe_trainData.npy',jaffe_trainData);
	np.save('data/jaffe_testData.npy',jaffe_testData)
	np.save('data/jaffe_trainLabels.npy',jaffe_trainLabels)
	np.save('data/jaffe_testLabels.npy',jaffe_testLabels)




source_path_jaffe = "/media/batman/New Volume/piyush/consortium/iitp/8th sem/EmotionRecog/Dataset/JapeneseWomenDataset/jaffe/"
read_jaffe_dataset(source_path_jaffe)

# Thanks Ramiz Raja for his: https://github.com/informramiz/opencv-face-recognition-python

from IPython.display import display
from PIL import Image
import sys, os
import numpy as np
import cv2

class FaceClassifier:
    faceSize = -1
    face_recognizer = None
    debug = False
    subjects = []
    
    def __init__(self, debug=False, face_resize=-1):
        #self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
        self.debug = debug
        self.faceSize = face_resize
        
    def train(self, train_path, delete_originals=False, show_train_images=False):
        print("Preparing data...")
        faces, labels = self.prepare_training_data(train_path, show_train_images=show_train_images)
        print("Data prepared")

        #print total faces and labels
        print("Total faces: ", len(faces))
        print("Labels tagged: ", len(labels), "with", len(np.unique(labels)), "labels")
        print("Training...")
        self.face_recognizer.train(faces, np.array(labels))
        print("We are ready!")

    def predict(self, test_path):
        ret = []
        detections = self.detect_face(test_path);
        for detection in detections:
            face, rect = detection[0], detection[1]
            if face is None:
                return "No_face"

            #predict the image using our face recognizer 
            label = self.face_recognizer.predict(face)
            #get name of respective label returned by face recognizer
            ret.append(self.subjects[label[0]-1])
        return ret
                
    def detect_face(self, img, grayscale_output=True):
        ret = []
        img = cv2.imread(img)
        if img is None:
            ret.append([None, None])
            return ret
        
        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #load OpenCV face detector, I am using LBP which is fast
        #there is also a more accurate but slow Haar classifier
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        #face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

        #let's detect multiscale (some images may be closer to camera than others) images
        #result is a list of faces
        faces = face_cascade.detectMultiScale(gray)#, scaleFactor=1.2, minNeighbors=5);

        #if no faces are detected then return original img
        if (len(faces) == 0):
            ret.append([None, None])
            return ret
        
        i = 0
        for face in faces:
            (x, y, w, h) = face
            
            #return only the face part of the image
            if (grayscale_output):
                img_face = gray[y:y+w, x:x+h]
            else:
                img_face = img[y:y+w, x:x+h]
                img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)

            if self.faceSize > 0:
                img_face = cv2.resize(img_face, (self.faceSize, self.faceSize), interpolation = cv2.INTER_AREA)
            ret.append([img_face, faces[i]])
        return ret
                
    def prepare_training_data(self, train_path, show_train_images=False):
        #------STEP-1--------
        #get the directories (one directory for each subject) in data folder
        dirs = os.listdir(train_path)
        
        #list to hold all subject faces
        faces = []
        #list to hold labels for all subjects
        labels = []
        self.subjects = []

        #let's go through each directory and read images within it
        label = 0
        for dir_name in dirs:
            #------STEP-2--------
            #extract label number of subject from dir_name
            #format of dir name = slabel
            #, so removing letter 's' from dir_name will give us label
            label += 1
            self.subjects.append(dir_name)
            print (dir_name, "label = ", label)

            #build path of directory containin images for current subject subject
            #sample subject_dir_path = "training-data/s1"
            subject_dir_path = train_path + "/" + dir_name

            #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            #------STEP-3--------
            #go through each image name, read image, 
            #detect face and add face to list of faces
            for image_name in subject_images_names:

                #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;

                #build image path
                #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                #detect face
                detections = self.detect_face(image_path);
                face, rect = detections[0][0], detections[0][1]

                #------STEP-4--------
                #for the purpose of this tutorial
                #we will ignore faces that are not detected
                if face is not None:
                    if show_train_images:
                        print ("Image: ", image_path)
                        display(Image.fromarray(face))
            
                    faces.append(face)
                    labels.append(label)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        return faces, labels
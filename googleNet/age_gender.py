import cv2
from imutils import paths
from os.path import dirname, abspath
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",
                default="res10_300x300_ssd_iter_140000.caffemodel",
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

print("[INFO] loading age model...")
ageNet = cv2.dnn.readNet(ageModel, ageProto)
print("[INFO] loading gender model...")
genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

path = dirname(dirname(abspath(__file__))) + "\\assets\\age_gender"
imagePaths = sorted(list(paths.list_images(path)))
#print(imagePaths)

for j, image_path in enumerate(imagePaths):
    image = cv2.imread(imagePaths[j])
    h, w, channels = image.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            #cv2.imshow("face"+str(j)+str(i), face)

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(face_blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(face_blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            text = "{},{}".format(gender, age)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    name = image_path.split("\\")[-1]
    cv2.imshow(name, image)

cv2.waitKey(0)
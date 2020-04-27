#!/usr/bin/env python2
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import re
import pickle
import sys

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface
from sklearn.mixture import GMM


video_name = '../video/test (4).mp4'

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    # print (reps)
    return (reps,bb)


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer
    #
    if img is None:
        return (None, None)
    repsAndBBs = getRep(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print ("No Face detected")
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print (predictions)
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences ,bbs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    dlibFacePredictor = os.path.join(
        dlibModelDir,
        "shape_predictor_68_face_landmarks.dat")
    
    networkModel = os.path.join(
        openfaceModelDir,
        'nn4.small2.v1.t7')

    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    
    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    start = time.time()

    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(networkModel, imgDim=args.imgDim, cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()


    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(video_name)
    video_capture.set(3, 480)
    video_capture.set(4, 640)
    if video_capture.isOpened() == False :
        print("no camera")
        exit()

    #==============================================================
    width = video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)

    # fourcc = cv2.CV_VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    filename = re.sub(r'\.mp4|avi|MOV$', '-recog.avi', video_name)

    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
    #======================================================

    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        print(ret, frame)
        persons, confidences, bbs = infer(frame, args)
        print ("P: " + str(persons) + " C: " + str(confidences))
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= 0.993 :#args.threshold:  # 0.5 is kept as threshold for known face.
                persons[i] = "Unknown"
        # for i, c in enumerate(confidences):
        #     if c <= 0.9 or persons[i] != "seoyoung":
        #         persons[i] = "Unknown"

        # Print the person name and conf value on the frame next to the person
        # Also print the bounding box
        for idx,person in enumerate(persons):
            x1, x2 = bbs[idx].left(), bbs[idx].right()
            y1, y2 = bbs[idx].top(), bbs[idx].bottom()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(frame, "{} @{:.2f}".format(person, confidences[idx]),
                        (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if person == "Unknown":
                face_img = frame[y1:y2, x1:x2]        
                face_img = cv2.medianBlur(face_img, 29)
                frame[y1:y2, x1:x2] = face_img

        # cv2.imshow('', frame)
        out.write(frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
        

#!/usr/bin/env python2
# -*- coding: utf-8 -*- 

import cv2
import sys
import numpy as np

import openface.helper

#얼국 인식용 xml 파일 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#전체 사진에서 얼굴 부위만 잘라 리턴
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #얼굴 찾기 
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #찾은 얼굴이 없으면 None으로 리턴 
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face


if __name__ == "__main__" :

    folder_name = sys.argv[1]
    trainCnt = int(sys.argv[2])

    # 추출한 얼굴 사진 저장할 폴더 정하기
    outputDir = '../training-images/'+folder_name+'/'
    openface.helper.mkdirP(outputDir)

    #카메라 실행 
    cap = cv2.VideoCapture(0)
    #저장할 이미지 카운트 변수 
    count = 0
    while True:
        ret, frame = cap.read()
        #얼굴 감지 하여 얼굴만 가져오기 
        if face_extractor(frame) is not None:
            count+=1
            face_tmp = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face_tmp, cv2.COLOR_BGR2GRAY)
            #faces폴더에 jpg파일로 저장 
            # ex > faces/user0.jpg   faces/user1.jpg ....
            file_name_path = outputDir+str(count)+'.jpg'         
            cv2.imwrite(file_name_path,face)
            
            # cv.putText(img, '글자', location, font, fontScale, color, thickness)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
            #
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==trainCnt:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')
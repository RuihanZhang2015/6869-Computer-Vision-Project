# -*- coding: utf-8 -*-
# @File       : LM.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-03 12:03
# @Description:

import cv2
import dlib
import numpy as np

import sys
from os.path import dirname, abspath

path = dirname(abspath(__file__))
print(path)
sys.path.append(path)


class Face_Landmark:
    def __init__(self):
        self.faceDetector = dlib.get_frontal_face_detector()
        self.landmarkPredictor = dlib.shape_predictor("E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\face_landmark\\shape_predictor_68_face_landmarks.dat")

    def get_landmark(self, img, visualize=False):
        height, width, channels = img.shape
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.faceDetector(grayImg, 1)
        points = []
        for face in faces:
            landmarks = self.landmarkPredictor(img, face)
            index = 0
            for pt in landmarks.parts():
                points.append({
                    "Index": index,
                    "X": pt.x,
                    "Y": pt.y
                })
        points = points[:17]
        if visualize:
            for point in points:
                cv2.circle(img, (point['X'], point['Y']),5,(0,255,0),4)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points

    def __hdistance(self, pt1, pt2):
        return abs(pt1['X']-pt2['X'])

    def __vdistance(self, pt1, pt2):
        return abs(pt1['Y'] - pt2['Y'])

    def calculate_feature(self, points):
        feature_vector = []
        distance_1 = self.__hdistance(points[0], points[16])
        distance_2 = self.__hdistance(points[1], points[15])
        distance_3 = self.__hdistance(points[2], points[14])
        distance_4 = self.__hdistance(points[3], points[13])
        distance_5 = self.__hdistance(points[4], points[12])
        distance_6 = self.__hdistance(points[5], points[11])
        distance_7 = self.__hdistance(points[6], points[10])
        distance_8 = self.__hdistance(points[7], points[9])
        distance_9 = self.__vdistance(points[0], points[8])
        distance_10 = self.__vdistance(points[2], points[8])
        distance_11 = self.__vdistance(points[4], points[8])
        distance_12 = self.__vdistance(points[6], points[8])
        feature_vector.append(distance_1)
        feature_vector.append(distance_2)
        feature_vector.append(distance_3)
        feature_vector.append(distance_4)
        feature_vector.append(distance_5)
        feature_vector.append(distance_6)
        feature_vector.append(distance_7)
        feature_vector.append(distance_8)
        feature_vector.append(distance_9)
        feature_vector.append(distance_10)
        feature_vector.append(distance_11)
        feature_vector.append(distance_12)
        feature_vector = np.array(feature_vector)
        feature_vector = feature_vector / max(feature_vector)
        return feature_vector


if __name__ == "__main__":
    face_landmark = Face_Landmark()
    img = cv2.imread("./mengke_2.jpg")
    marks = face_landmark.get_landmark(img, True)
    features = face_landmark.calculate_feature(marks)
    print(features)

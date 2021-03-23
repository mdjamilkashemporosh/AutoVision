import cv2
from random import randrange

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

cap = cv2.VideoCapture("cars2.mp4")

while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w+5,y+h+5),(randrange(256),randrange(256),randrange(256)),3)

    cv2.imshow('Car Detector',img)
    cv2.waitKey(30)



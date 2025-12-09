import cv2
import numpy as np

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')

img = cv2.imread('face1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 加载人脸检测器
face_detect = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
faces = face_detect.detectMultiScale(gray)


CONFIDENCE_THRESHOLD = 90

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 进行人脸识别
    id, confidence = recogizer.predict(gray[y:y + h, x:x + w])
    print('标签id：', id, '置信评分：', confidence)

    # 判断是否为未知人脸
    if confidence < CONFIDENCE_THRESHOLD:
        # 已知人脸，显示ID
        label = f"ID: {id}"

    else:
        # 未知人脸
        label = "Unknown"
        print('Unknown')
    # 在矩形框上方显示标签
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
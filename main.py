import cv2
import numpy as np

frame_width = 640
frame_height = 480
faceProto = "ModalDirectory/deploy.prototxt.txt"
faceModel = "ModalDirectory/res10_300x300_ssd_iter_140000_fp16.caffemodel"
conf_threshold = 0.5
genderProto = "ModalDirectory/deploy_gender.prototxt"
genderModel = "ModalDirectory/gender_net.caffemodel"
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageProto = "ModalDirectory/deploy_age.prototxt"
ageModel = "ModalDirectory/age_net.caffemodel"
ageList = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

def detectFaces(frame, net, conf_threshold=0.5):
    # Отправляем кадр для обработки в нейросети
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    #Получаем результат
    detections = net.forward()
    faces = []
    #Цикл для нахождения координат лица на кадре
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces

def videoDisplay():
    # Загружаем модели для нейросети
    face_net = cv2.dnn.readNetFromCaffe(faceProto, faceModel)
    genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
    ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
    # Захватываем изображение с веб-камеры
    vebCamVideo = cv2.VideoCapture(0)
    # Вывод изображения, пока не нажата любая клавиша
    while cv2.waitKey(1) < 0:
        hasFrame, frame = vebCamVideo.read()
        if not hasFrame:
            cv2.waitKey()
            break
        # Обнаружение лица
        faces = detectFaces(frame, face_net)
        # Вывод прямоугольника вокруг лица
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color = (255, 0, 0), thickness= 2)
        # Определение пола
        genderRecognition(frame, faces, genderNet)
        # Определение возраста
        ageRecognition(frame, faces, ageNet)
        cv2.imshow("Face and age recognition", frame)

def genderRecognition(frame, faceBoxes, net):
    #Цикл для всех обнаруженых лиц
    for faceBox in faceBoxes:
        # Обрезаем лицо по координатам
        face = frame[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]]
        # Отправляем лицо для обработки в нейросети
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        net.setInput(blob)
        # Получаем результат
        genderPred = net.forward()
        gender = genderList[genderPred[0].argmax()]
        # Выводим полученный пол
        cv2.putText(frame, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

def ageRecognition(frame, faceBoxes, net):
    #Цикл для всех обнаруженых лиц
    for faceBox in faceBoxes:
        # Обрезаем лицо по координатам
        face = frame[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]]
        # Отправляем лицо для обработки в нейросети
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        net.setInput(blob)
        # Получаем результат
        agePred = net.forward()
        age = ageList[agePred[0].argmax()]
        # Выводим полученный возраст
        cv2.putText(frame, f'{age}', (faceBox[0] + 80, faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

videoDisplay()

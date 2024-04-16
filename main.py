import cv2
import numpy as np

frame_width = 640
frame_height = 480
faceProto = "ModalDirectory/deploy.prototxt.txt"
faceModel = "ModalDirectory/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(faceProto, faceModel)
confidence_threshold = 0.5
genderProto = "ModalDirectory/deploy_gender.prototxt"
genderModel = "ModalDirectory/gender_net.caffemodel"
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)


def videoDisplay():
    # Захватываем видео с камеры
    vebCamVideo = cv2.VideoCapture(0)
    # Цикл вывода изображения камеры пока не нажата клавиша
    while cv2.waitKey(1) < 0:
        hasFrame, vebCamVideoFrame = vebCamVideo.read()
        # Если нет изображения программа прекратит свое выполнение
        if not hasFrame:
            cv2.waitKey()
            break
        faces = faceDetection(vebCamVideoFrame)
        genderRecognition(vebCamVideoFrame, faces)
        # Вывод видео с камеры
        cv2.imshow("Name", vebCamVideoFrame)


def faceDetection(frame):
    # Преобразуем кадр в массив двоичных данных (blob) для ввода в нейронной сети (NN)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # Установим картинку в качестве входа NN
    face_net.setInput(blob)
    # Получаем прогноз и позываем его
    output = np.squeeze(face_net.forward())
    # Список для записи результата
    faces = []
    # Перебираем найденные лица
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            # Преобразуем в целые числа
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - \
                                             10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color = (255, 0, 0), thickness= 2)
        return faces

def genderRecognition(frame, faceBoxes):
    for faceBox in faceBoxes:
        face = frame[max(0,faceBox[1]):
                   min(faceBox[3],frame.shape[0]-1),max(0,faceBox[0])
                   :min(faceBox[2], frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (frame_width, frame_height), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = np.squeeze(genderNet.getLayerNames())
        gender = genderList[genderPred[0].argmax()]
        cv2.putText(frame, f'{gender}',(faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)


videoDisplay()

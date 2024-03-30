import cv2


def videoDisplay():
    # Захватываем видео с камеры
    vebCamVideo = cv2.VideoCapture(0)
    # Цикл вывода изображения камеры пока не нажата клавиша
    while cv2.waitKey(1) < 0:
        hasFrame, vebCamVideoFrame = vebCamVideo.read()
        #Если нет изображения программа прекратит свое выполнение
        if not hasFrame:
            cv2.waitKey()
            break
        vebCamVideoFrame = faceDetection(vebCamVideoFrame)
        #Вывод видео с камеры
        cv2.imshow("Name", vebCamVideoFrame)

def faceDetection(frame):
    #Преобразуем считываемое изображение к оттенкам серого
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Инициализируем распознователь лиц
    faceCascade = cv2.CascadeClassifier("ModalDirectory/haarcascade_frontalface_default.xml")
    #Обнаружение лиц на изображении
    face = faceCascade.detectMultiScale(frameGray)
    for x, y, width, height in face:
        cv2.rectangle(frame, (x, y), (width + x, height + y), color = (255, 0, 0), thickness= 2)
    return frame

videoDisplay()

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
        #Вывод видео с камеры
        cv2.imshow("Name", vebCamVideoFrame)


videoDisplay()
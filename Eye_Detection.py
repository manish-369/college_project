import cv2

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Getting corners around the eyes
    # 1.3 = scale factor, 5 = minimum neighbor
    eyes = eyeCascade.detectMultiScale(imgGray, 1.3, 5)

    # drawing bounding box around eyes
    for (ex, ey, ew, eh) in eyes:
        img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')

import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    # kameradan alınan görüntüleri "frame"e kayıt eder.
    ret, frame = cap.read()

    # ayna etkisi yapmak için.
    frame = cv2.flip(frame, 1)

    frame = cv2.medianBlur(frame, 5)

    # BGR dan HSV formatına çeviriyoruz.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv formatında kırmızı rengin alt ve üst sınırları.
    lower_red = np.array([161,155,84])
    upper_red = np.array([179,255,255])

    # rengi maskeleme işlemi.
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # kırmızı renge maskeleme işlemi.
    red_red_mask = cv2.bitwise_and(frame, frame, mask = red_mask)

    dilate = cv2.dilate(red_mask, (3,3), iterations=3)

    contour, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contour)):
        cv2.drawContours(frame, contour, i, (0,255,255), 4)

    # kameradan alınan görüntüleri ekrana basmaya yarıyor.
    cv2.imshow("Camera", frame)
    # maskelenmiş kamera görüntüsü.
    cv2.imshow("Mask Camera", red_mask)
    # kırmızı maskelenmiş kamera görüntüsü.
    cv2.imshow("Red Mask", red_red_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

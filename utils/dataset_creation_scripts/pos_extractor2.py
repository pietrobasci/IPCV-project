import os
import cv2


images = list(sorted(i for i in os.listdir("dataset1/images")
                                 if i.lower().endswith(('.png', '.jpg', '.jpeg'))))
info_file = open("output/info.dat","a")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


n_img = 3432

for img in images:
    img = cv2.imread("dataset1/images/" + img)
    h_orig, w_orig, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minSize=(int(0.25*w_orig), int(0.25*h_orig)))

    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]

    if w < 20 or h < 20:
        continue

    img_crop = None

    if w > h:
        dx = int((w - h) / 2)
        img_crop = img[max(y-dx, 0):min(y+h+dx, h_orig), x:x+w]
    elif w < h:
        dx = int((h - w) / 2)
        img_crop = img[y:y+h, max(x-dx, 0):min(x+w+dx, w_orig)]
    else:
        #img_crop = img[y:y+h, x:x+w]
        img_crop = img[max(y-int(0.1*h), 0):min(y+h+int(0.1*h), h_orig), max(x-int(0.1*w), 0):min(x+w+int(0.1*w), w_orig)]

    img_crop = cv2.resize(img_crop, (100, 100))
    out_file = "output/pos/img"+str(n_img)+".png"
    cv2.imwrite(out_file, img_crop)
    info_file.write("pos/img"+str(n_img)+".png 1 0 0 100 100\n")
    n_img += 1

info_file.close()

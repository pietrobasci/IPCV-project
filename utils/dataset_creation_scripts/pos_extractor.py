import os
import cv2
import numpy as np
import xmltodict


images = list(sorted(i for i in os.listdir("dataset/images")
                                 if i.lower().endswith(('.png', '.jpg', '.jpeg'))))
annotations = list(sorted(a for a in os.listdir("dataset/annotations")
                                        if a.lower().endswith(('.xml'))))
info_file = open("output/info.dat","w")
n_img = 0

for (img, ann) in zip(images, annotations):
    img = cv2.imread("dataset/images/" + img)
    x = xmltodict.parse(open("dataset/annotations/" + ann, 'rb'))
    h_orig, w_orig, _ = img.shape

    objects = []

    if isinstance(x['annotation']['object'], list):
        objects = x['annotation']['object']
    else:
        objects.append(x['annotation']['object'])

    for obj in objects:
        if obj['name'] == "with_mask":
            bndbox = obj['bndbox']
            bndbox = np.array([int(bndbox['xmin']), int(bndbox['ymin']),
                               int(bndbox['xmax']), int(bndbox['ymax'])])

            w = bndbox[2] - bndbox[0]
            h = bndbox[3] - bndbox[1]

            if w < 20 or h < 20:
                continue

            img_crop = None

            if w > h:
                dx = int((w - h) / 2)
                #img_crop = img[bndbox[1]-dx:bndbox[3]+dx, bndbox[0]:bndbox[2]]
                img_crop = img[max(bndbox[1]-dx, 0):min(bndbox[3]+dx, h_orig), bndbox[0]:bndbox[2]]

            else:
                dx = int((h - w) / 2)
                #img_crop = img[bndbox[1]:bndbox[3], bndbox[0]-dx:bndbox[2]+dx]
                img_crop = img[bndbox[1]:bndbox[3], max(bndbox[0]-dx, 0):min(bndbox[2]+dx, w_orig)]

            img_crop = cv2.resize(img_crop, (100, 100))
            out_file = "output/pos/img"+str(n_img)+".png"
            cv2.imwrite(out_file, img_crop)
            info_file.write("pos/img"+str(n_img)+".png 1 0 0 100 100\n")
            n_img += 1

info_file.close()

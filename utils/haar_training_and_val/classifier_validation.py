import os
import cv2

scaleFactor = 1.07
minSize = (75*3, 75*3)
tp, fp, tn, fn = 0, 0, 0, 0
path = "dataset/mask_detection_validationset/"

mask_cascade = cv2.CascadeClassifier('haarcascade_facemask_25s.xml')

# Test on Negative samples
neg_images = list(sorted(i for i in os.listdir(path + "neg")
                         if i.lower().endswith(('.png', '.jpg', '.jpeg'))))

for img in neg_images:
    img = cv2.imread(path + "neg/" + img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, None, fx=3, fy=3)

    faces = mask_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minSize=minSize)

    if len(faces) > 0:
        fp += 1
    else:
        tn += 1

# Test on Positive samples
pos_images = list(sorted(i for i in os.listdir(path + "pos")
                         if i.lower().endswith(('.png', '.jpg', '.jpeg'))))

for img in pos_images:
    img = cv2.imread(path + "pos/" + img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, None, fx=3, fy=3)

    faces = mask_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minSize=minSize)

    if len(faces) > 0:
        tp += 1
    else:
        fn += 1

# Compute metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)

# Print results
print("--Confusion Matrix--")
print("                   Actual Positive  |  Actual Negative")
print("Predicted Positive        {}       |         {}".format(tp, fp))
print("Predicted Negative        {}       |         {}".format(fn, tn))
print("-"*60)
print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("f1-score: {:.3f}".format(f1_score))

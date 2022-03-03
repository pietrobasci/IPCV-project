import cv2
import matplotlib.pyplot as plt
from people_detector import PeopleDetector
from centroid_tracker import CentroidTracker
from imutils.video import FPS


def grab_frame(cap, detector, tracker, face_cascade, mask_cascade):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :param detector: the people_detector object
    :param tracker: the centroid_tracker object
    :param face_cascade: the face classifier
    :param mask_cascade: the face mask classifier
    :return: the captured image and the foreground mask
    """
    ret, img = cap.read()
    # Apply background subtraction to find people (roi)
    mask, roi = detector.detect(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Array of int used for the majority vote (1=with_mask, -1=without_mask, 0=undefined)
    masks = [0]*len(roi)
    for i, (xp, yp, wp, hp) in enumerate(roi):
        # Extract roi
        region = gray[yp:yp+int(hp/2), xp:xp+wp]
        # Resize to improve detection
        region = cv2.resize(region, None, fx=3, fy=3)
        # Detect faces
        faces = face_cascade.detectMultiScale(region, scaleFactor=1.3, minSize=(90, 90))
        for (xf, yf, wf, hf) in faces:
            xf = int(xf / 3) + xp
            yf = int(yf / 3) + yp
            wf = int(wf / 3)
            hf = int(hf / 3)
            # Increase image crop of about 20%
            x0, y0, x1, y1 = increase_size(xf, yf, wf, hf, gray.shape)
            face = gray[y0:y1, x0:x1]
            face = cv2.resize(face, None, fx=3, fy=3)
            # Detect mask
            face_mask = mask_cascade.detectMultiScale(face, scaleFactor=1.07, minSize=(int(0.75*wf*3), int(0.75*hf*3)))
            # If found set the array value to 1 otherwise to -1
            if len(face_mask) > 0:
                #xm, ym, wm, hm = face_mask[0]
                #xm = int(xm / 3) + x0
                #ym = int(ym / 3) + y0
                #wm = int(wm / 3)
                #hm = int(hm / 3)
                masks[i] = 1

                #cv2.rectangle(img, (xm, ym), (xm + wm, ym + hm), (255, 0, 0), 2)
                cv2.rectangle(img, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)
                cv2.putText(img, "With Mask", (xf, yf + hf + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                masks[i] = -1
                cv2.rectangle(img, (xf, yf), (xf + wf, yf + hf), (0, 0, 255), 2)
                cv2.putText(img, "Without Mask", (xf, yf + hf + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(mask, (xp, yp), (xp + wp, yp + hp), (0, 255, 255), 2)

    # Update our centroid tracker using the computed set of bounding box rectangles
    objects = tracker.update(roi, masks)

    # Loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # Draw both the object ID and the centroid on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(mask, text, (centroid[0] - 15, centroid[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 150, 255), 3)
        cv2.circle(mask, (centroid[0], centroid[1]), 8, (80, 150, 255), -1)

    # Draw green lines
    cv2.line(img, (0, tracker.entranceLine), (mask.shape[1], tracker.entranceLine), (0, 255, 0), 2)
    cv2.line(mask, (0, tracker.entranceLine), (mask.shape[1], tracker.entranceLine), (0, 255, 0), 2)

    # Draw counters
    text = "People: {}".format(tracker.count)
    cv2.putText(img, text, (0, img.shape[0] - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    text = "Mask: {}".format(tracker.maskcount)
    cv2.putText(img, text, (0, img.shape[0] - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    text = "No-Mask: {}".format(tracker.nomaskcount)
    cv2.putText(img, text, (0, img.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    text = "Undefined: {}".format(tracker.undefined)
    cv2.putText(img, text, (0, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    return img, mask


def handle_close(event, cap, fps):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    :param fps: the FPS counter
    """
    fps.stop()
    cap.release()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def bgr_to_rgb(image):
    """
    Convert a BGR image into RBG
    :param image: the BGR image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def increase_size(x, y, w, h, shape):
    # increase the size of a given image crop
    x0 = max(0, x - int(0.2 * w))
    y0 = max(0, y + int(0.2 * h))
    x1 = min(shape[1], x + int(1.2 * w))
    y1 = min(shape[0], y + int(1.4 * h))
    return x0, y0, x1, y1


def main():
    # init the camera
    cap = cv2.VideoCapture(0)

    # enable Matplotlib interactive mode
    plt.ion()
    # create the FPS counter
    fps = FPS()

    # create a figure to be updated
    fig = plt.figure()
    # intercept the window's close event to call the handle_close() function
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap, fps))

    # prep a variable for the first run
    img1 = None
    img2 = None

    # prep People detector and Centroid tracker objects
    detector = PeopleDetector()
    tracker = CentroidTracker()

    # prep face cascade and mask cascade objects
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mask_cascade = cv2.CascadeClassifier('haarcascade_facemask_25s.xml')

    # start FPS counter
    fps.start()

    while cap.isOpened():
        # get the current frame
        frame, mask = grab_frame(cap, detector, tracker, face_cascade, mask_cascade)

        if img1 is None:
            # Frame
            plt.subplot(211)
            # convert it in RBG (for Matplotlib)
            img1 = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # Foreground mask
            plt.subplot(212)
            # convert it in RBG (for Matplotlib)
            img2 = plt.imshow(bgr_to_rgb(mask))
            plt.axis("off")  # hide axis, ticks, ...

            # show the plot!
            plt.tight_layout(pad=0.1)
            plt.show()
        else:
            # set the current frame as the data to show
            img1.set_data(bgr_to_rgb(frame))
            img2.set_data(bgr_to_rgb(mask))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            plt.pause(1/30)  # pause: 30 frames per second

        # update FPS counter
        fps.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)

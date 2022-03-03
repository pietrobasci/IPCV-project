import cv2
import matplotlib.pyplot as plt


def grab_frame(cap, face_cascade, mask_cascade):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :param face_cascade: the face classifier
    :param mask_cascade: the face mask classifier
    :return: the captured image
    """
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minSize=(90, 90))

    for (x, y, w, h) in faces:
        x0, y0, x1, y1 = increase_size(x, y, w, h, gray.shape)
        face = gray[y0:y1, x0:x1]
        face = cv2.resize(face, None, fx=3, fy=3)

        mask = mask_cascade.detectMultiScale(face, scaleFactor=1.07, minSize=(int(0.75*w*3), int(0.75*h*3)))

        if len(mask) > 0:
            xm, ym, wm, hm = mask[0]
            xm = int(xm / 3) + x0
            ym = int(ym / 3) + y0
            wm = int(wm / 3)
            hm = int(hm / 3)

            cv2.rectangle(img, (xm, ym), (xm + wm, ym + hm), (255, 0, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "With Mask", (x, y + h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Without Mask", (x, y + h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return img


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


def bgr_to_rgb(image):
    """
    Convert a BGR image into RBG
    :param image: the BGR image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def increase_size(x, y, w, h, shape):
    """
    Increase the size of a given image
    """
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

    # create a figure to be updated
    fig = plt.figure()
    # intercept the window's close event to call the handle_close() function
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    img = None

    # prep face cascade object
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mask_cascade = cv2.CascadeClassifier('haarcascade_facemask_25s.xml')

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap, face_cascade, mask_cascade)

        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # set the current frame as the data to show
            img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            plt.pause(1/30)  # pause: 30 frames per second


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)

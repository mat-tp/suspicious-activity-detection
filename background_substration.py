# import cv2
# import numpy as np

# # Load the video
# cap = cv2.VideoCapture("surveillanceVideosDataset/surveillanceVideos/3982ExFo_IndPO_HQ_C3.mp4")


# _, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

# while True:
#     _, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Removing the Noise in the Vedio
#     gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
#     # We can try markov Transformation to delete the white noise...


#     # difference = cv2.absdiff(first_frame,frame)
#     difference = cv2.absdiff(first_gray,gray_frame)
#     ret,difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)    

#     cv2.imshow("First Frame", first_frame)
#     cv2.imshow("Frame", first_frame)
#     cv2.imshow("difference",difference)

#     key = cv2.waitKey(30)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("surveillanceVideosDataset/surveillanceVideos/3982ExFo_IndPO_HQ_C3.mp4")

# Adapts to the lighting changes during the vedio.Encoporates Guassian Filtering and Transformation to remove the noise.
# Detection of the shadows
subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)

while True:
    _, frame = cap.read()
    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

import json
import cv2

with open("activity_videos.json") as f:
    data = json.load(f)

for activity, videos in data.items():
    print(activity, ":", len(videos))

for activity, v in data.items():
    video = "surveillanceVideos/" + data[activity][0]
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    if ret:
        cv2.imshow(activity, frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
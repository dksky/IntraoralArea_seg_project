import cv2

# 1.2视频读取
# 打开内置摄像头
cap = cv2.VideoCapture(0)
# 打开视频
#cap = cv2.VideoCapture('why.mp4')
# 或者视频每秒多少帧的数据
fps = cap.get(5)
i = 0
while (True):
    # 读取一帧
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 480))
    cv2.imshow("camera", frame)
    # 读完之后结束退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destoryAllWindows()
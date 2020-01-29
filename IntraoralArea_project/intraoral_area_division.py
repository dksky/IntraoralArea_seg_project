import cv2
import time
import predict as predict
from keras.models import load_model

model = load_model("intraoralArea.hdf5")

def detect():
    # 创建人脸检测的对象
    face_cascade = cv2.CascadeClassifier("E:/deep_learning/Anaconda3/envs/tensorFlow_v1.5/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    # 创建眼睛检测的对象
    eye_cascade = cv2.CascadeClassifier("E:/deep_learning/Anaconda3/envs/tensorFlow_v1.5/lib/site-packages/cv2/data/haarcascade_eye.xml")
    # 连接摄像头的对象，0表示摄像头的编号
    camera = cv2.VideoCapture(0)

    t = time.time()
    num = 0;
    while True:
        if time.time()-t < 1:
            num=num+1
        else :
            print("当前fps="+str(num))
            num=0
            t = time.time()

        # 读取当前帧
        ret, frame = camera.read()

        predictValue = predict.getPredictValueByImg(frame, model)

        cv2.imshow("camera", predictValue/255)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break;

    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    detect()
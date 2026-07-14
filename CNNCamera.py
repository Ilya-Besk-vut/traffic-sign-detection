#from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras import models
import numpy
import cv2
from skimage import exposure

model = tensorflow.keras.models.load_model("Best_Sign_Class.keras")

print("NET LOADED")
model.summary()

ESCAPE = 27
key = 1

# Sign = {(1, 0, 0, 0, 0, 0, 0): "Main road",
#         (0, 1, 0, 0, 0, 0, 0): "Give way",
#         (0, 0, 1, 0, 0, 0, 0): "Stop",
#         (0, 0, 0, 1, 0, 0, 0): "Traffic is prohibited",
#         (0, 0, 0, 0, 1, 0, 0): "entry is forbidden",
#         (0, 0, 0, 0, 0, 1, 0): "Rough road",
#         (0, 0, 0, 0, 0, 0, 1): "RoadWork"}

target_names = ["Main road", "Give way", "Stop", "Traffic is prohibited", "entry is forbidden", "Rough road", "RoadWork"]

cap = cv2.VideoCapture(0)

while (key != ESCAPE):
    ret, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]
    shift = (width - height) // 2

    cv2.rectangle(frame, (shift, 0), (width-shift, height), (50, 250, 50), 2)
    key = cv2.waitKey(1)

    cuttedFrame = frame[:, shift:width-shift]
    datas = cv2.resize(cuttedFrame, (32, 32))
    datas = exposure.equalize_adapthist(datas, clip_limit=0.1)

    datas = datas.astype("float") /255
    datas = numpy.expand_dims(datas,axis=0)

    pred = model.predict(datas)
    pred = pred[0]

    numname = numpy.argmax(pred)
    frame = cv2.putText(frame, target_names[numname], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 4)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
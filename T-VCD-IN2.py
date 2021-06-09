import cv2
import numpy as np
import time
import winsound

#INBUILT TRAINING
net = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
#net = cv2.dnn.readNet('yolov3.weights','yolo3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()


#CUSTOM TRAINING
# net = cv2.dnn.readNet("yolov3_training_last.weights","yolov3_testing.cfg")
# classes = ["car"]


#LANE POINTS
x1 = 415
x2 = 490
x3 = 644
x4 = 177
y1 = 383
y2 = 530
ym = (y1+y2)/2

tpo = 0
tpr = 0




#cap=cv2.VideoCapture("Resources/AUH roads")
cap=cv2.VideoCapture("Resources/AUH roads2.mp4")


#ROI
def roi(x,y,img):

    m1 = (y2-y1)/(x4-x1)
    m2 = (y2-y1)/(x3-x2)
    if (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= ym and y < y2):
        return 2
    elif (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= y1 and y < ym):
        return 1
    else:
        return 0

#PLAY SOUND

def playSoundOrange(tpr):
    t = time.time()
    if(t-tpr>10):
        winsound.PlaySound("single", winsound.SND_FILENAME)

        tpr = t
    tpr = t
    return tpr

def playSoundRed(tpo):
    t = time.time()
    if(t-tpo>10):
        winsound.PlaySound("single", winsound.SND_FILENAME)
        winsound.PlaySound("double", winsound.SND_FILENAME)
        tpo = t
    return tpo



c=0
while True:
    if(c%10!=0):
        _, img = cap.read()

        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        c+=1

        continue
    c+=1
    #print(c)
    _, img = cap.read()


    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    #print(height, width)
    #432,768
    #roi = img[0:432,225:650]
    pts = np.array([[x4,y2],[x1,y1],[x2,y1],[x3,y2]])
    cv2.polylines(img,[pts],True,(0,255,255),2)


    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    class_ids = []
    confidences = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # obj detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width+20)
                h = int(detection[3] * height+20)

                # cv2.circle(img,(center_x, center_y),10,(0,0,0),3)
                # Rectangle Coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if (len(indexes) > 0):
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if(w*h<200000):
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if (roi(x+w/2,y+h,img)==2):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Careful",(x,y+20),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                    tpr = playSoundRed(tpr)
                elif(roi(x+w/2,y+h,img)==1):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(img, " " , (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                    tpo = playSoundOrange(tpo)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, " " , (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow('Image',img)
    #cv2.imshow('ROI', roi)
    key = cv2.waitKey(200)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("emotion_model.h5")

labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

#face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
) #Using OpenCV’s built-in haarcascade path

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1,48,48,1))

        pred = model.predict(roi)[0]
        emotion = labels[np.argmax(pred)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

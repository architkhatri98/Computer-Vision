import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading haarcascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #making faces object
    for (x,y,w,h) in faces: #x,y are left-top corner and w,h are width and height respectively
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[x:x+w,y:y+h] #region of interest
        roi_color = frame[x:x+w,y:y+h]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) #making eyes object
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    # this frame object contains all the rectangles applied on the continous stream of video images
    return frame


video_capture = cv2.VideoCapture(0) #0 for internal webcam
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color image to gray
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas) #show the canvas stream as 'Video'
    if cv2.waitKey(1) & 0xFF == ord('q'): #pressing q will quit the program
        break

video_capture.release()
cv2.destroyAllWindows()

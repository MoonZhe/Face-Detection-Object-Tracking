import cv2

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
fullbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_fullbody.xml")
upperbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_upperbody.xml")
lowerbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_lowerbody.xml")

cam = cv2.VideoCapture(0)
#video = cv2.VideoCapture("test/Koay.mp4")
#if video.isOpened():
#    print "koay.mp4"
#else:
#    print "video read failed"
#while (video.isOpened()):
counter = 0
while cam.isOpened():
    ok, img = cam.read()
    #img1 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    #ok, img = video.read()
    #if ok:
    #    print "ok"
    #else:
    #     print "koay.mp4"   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #full_body = fullbody_cascade.detectMultiScale(gray)
    #upperbody = upperbody_cascade.detectMultiScale(gray)
    #lowerbody = lowerbody_cascade.detectMultiScale(gray)
    if not ok: 
            break
   
    hello = []

    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        hello.append(roi_color)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        if (counter == 5):
            for x in hello:
                    k = cv2.waitKey(1) & 0xFF
                    del hello[:]
                    if k == ord('q'):
                     #cv2.destroyWindow("Recognised Face0")
                     #cv2.destroyWindow("Recognised Face1")  
                     #cv2.destroyWindow("Recognised Face2")  
                     #cv2.destroyWindow("Recognised Face3")  
                     #cv2.destroyWindow("Recognised Face4")  
                     print "retake"
                     counter = 0
                     
        if (counter <= 4):   
           #cv2.imshow("Recognised Face" + str(counter), roi_color)
            cv2.imwrite('output/face_color' + str(counter) +'.png', roi_color)
            counter = counter + 1
             
   
    cv2.putText(img, "Number of face detected: " + str(int(len(faces))), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.imshow('Face Detection',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
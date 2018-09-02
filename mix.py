import cv2
import numpy as np

colour = input ("What colour do you want to track? \nEnter 1 for red, 2 for blue, 3 for green, 4 for orange, 5 for yellow\n")

while (colour <1 or colour > 5):
    colour = input("Invalid number. Please enter again: ")

if (colour == 1):
    lowerBound = np.array([166, 84, 141])
    upperBound = np.array([186, 255, 255])
        
elif (colour == 2):
    lowerBound=np.array([97,100,117])
    upperBound=np.array([117,255,255])
    
elif (colour == 3):
    lowerBound=np.array([0,128,0])
    upperBound=np.array([124,252,0])

elif (colour == 4):    
    lowerBound=np.array([0,50,80])
    upperBound=np.array([20,255,255])
    
elif (colour == 5):
    lowerBound=np.array([23,59,119])
    upperBound=np.array([54,255,255])



lowerBound1 = lowerBound
upperBound1 = upperBound

print "Press Q to retake another 5 photos"

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
fullbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_fullbody.xml")
upperbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_upperbody.xml")
lowerbody_cascade = cv2.CascadeClassifier("cascade/haarcascade_lowerbody.xml")

cam = cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
#video = cv2.VideoCapture("../test/Koay.avi")
counter = 0
#while (video.isOpened()):

while cam.isOpened():
   
    ok, img = cam.read()
    #ok, img = video.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #full_body = fullbody_cascade.detectMultiScale(gray)
    #upperbody = upperbody_cascade.detectMultiScale(gray)
    #lowerbody = lowerbody_cascade.detectMultiScale(gray)
    if not ok: 
            break
    
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound1,upperBound1)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    #cv2.imshow("cam",img)   
    
    
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
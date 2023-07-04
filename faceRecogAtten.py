import  face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


video_capture= cv2.VideoCapture(0)#0 is the camera num we use..

#loading known faces:

stark_image = face_recognition.load_image_file("faces/stark.jpg")
stark_encoding=face_recognition.face_encodings(stark_image)[0]#here 0 means we need the first image from the returned list of faces in any image

vikash_image = face_recognition.load_image_file("faces/vikash.jpg")
vikash_encoding=face_recognition.face_encodings(vikash_image)[0]

SteveRogers_image = face_recognition.load_image_file("faces/captain america.jpeg")
SteveRogers_encoding=face_recognition.face_encodings(SteveRogers_image)[0]

mom_image = face_recognition.load_image_file("faces/mom.jpg")
mom_encoding=face_recognition.face_encodings(mom_image)[0]

daddy_image = face_recognition.load_image_file("faces/daddy.jpg")
daddy_encoding=face_recognition.face_encodings(daddy_image)[0]

all_face_encoding=[vikash_encoding,SteveRogers_encoding,stark_encoding,daddy_encoding,mom_encoding]
all_face_names=["vikash","Captain America","Iron Man","daddy","mom"]

#all students in class:
students=all_face_names.copy()

face_location=[]
face_encodings=[]

#Get current date and time

now=datetime.now()

#to change the formate of date and time:

current_date=now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv","w+",newline="")#f is a variable that will store the file object after opening the file.
lnwriter=csv.writer(f)#csv.writer() is a function within the csv module that returns a writer object, which is used for writing CSV data to a file.


while True:
    _,frame=video_capture.read()#_=as the first argument is if the video captured was successfull and frame=then second arg is frame.
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recognizing faces..
    face_location   =face_recognition.face_locations(rgb_small_frame)
    face_encodings =face_recognition.face_encodings(rgb_small_frame,face_location)

    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(all_face_encoding,face_encoding)
        face_distance=face_recognition.face_distance(all_face_encoding,face_encoding)#says how similar is your face to the face that is known already..
        best_match_index=np.argmin(face_distance)#stores the best match by storign the minimum distance..

        if(matches[best_match_index]):#matches has the true,false value if faces are wether matched or not..
            name=all_face_names[best_match_index]

        #adding text if person is present:
        if name in all_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText=(10,100)
            fontScale = 1.5
            fontColor=(255,0,0)
            thickness=3
            lineType=2
            cv2.putText(frame,name+"  Present ",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)
            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M%S")
                lnwriter.writerow([name,current_time])





    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()










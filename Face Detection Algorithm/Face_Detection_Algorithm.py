import cv2 as cv

#CascadeClassifier Object to enable algorithm recognize face
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

#Read the image
img = cv.imread("group faces.jpg")

#Convert to grayscale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Search the coordinates of image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

#Rectangles to mark identified faces
for x,y,w,h in faces:
    img = cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

#Resize the image to display
resized = cv.resize(img, (int(img.shape[0]/2), int(img.shape[1]/5)))

#Display the image
cv.imshow("Group Faces", resized)

cv.waitKey(0)

cv.destroyAllWindows()
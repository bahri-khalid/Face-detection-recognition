import cv2 as cv
import face_recognition
from face_recognition import face_locations as fl
from PIL import Image, ImageDraw
import numpy as np
#capture = cv.VideoCapture("../../Videos/demoAnimation.mp4")
capture = cv.VideoCapture(0)

khalid_image = face_recognition.load_image_file("khalid.jpg") # use your personal image here for training 
khalid_image = cv.cvtColor(khalid_image,cv.COLOR_BGR2RGB)
khalid_face_encoding = face_recognition.face_encodings(khalid_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    khalid_face_encoding,
]
known_face_names = [
    "khalid bahri"
]


while True:
    isTrue, unknown_image = capture.read()
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            draw.rectangle(((left, top), (right, bottom)), fill=(0,0,0),outline=(0, 0, 255))

        # Draw a box around the face using the Pillow module

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 50), (right, bottom)), fill=(255, 255, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
    del draw
    cv.imshow("hello again",np.array(pil_image))
    if cv.waitKey(20) &  0xFF == ord("c"):
 	    break


capture.release()
cv.distroyAllWindows()
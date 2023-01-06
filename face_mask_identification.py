import cv2
import numpy as np
import tensorflow as tf  # tensorflow version 2.2 and above

# Download the trained model and save it in model folder
model = tf.keras.models.load_model(filepath="final_inception_face_mask_detection.h5")


target_shape=(160, 160)  # This shape is required by Object Classification model
cap = cv2.VideoCapture(0)  # Before starting this, make sure you have camera on your machine
out = cv2.VideoWriter(
    'output_haar.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))
while True:
    # Identify the frame from video file
    r, frame = cap.read()
    # Haar cascade requires gray scale images, so we need to convert the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Download haarcascade xml file from resources or from below location
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml
    # Load Haar Cascade xml for human face deduction
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # Pass gray image to Classifier for identification of face,
    # it returns the co-ordinates of face area in terms of x, y, w, h
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    cordinates = []
    images_faces = []
    face_label = []
    for (x, y, w, h) in faces:
        try:
            startX = int(x)
            startY = int(y)
            endX = int((x + w))
            endY = int((y + h))
            # Cut the face from original non-gray image and record it array
            if startX != endX and startY != endY:
                face_image = frame[startY:endY, startX:endX]
                cordinates.append([startX, startY, endX, endY])
                images_faces.append(face_image)
                # Pre process/resize input image in accordance to trained model
                height, width, _ = face_image.shape
                image_resized = cv2.resize(face_image, target_shape)
                image_np = image_resized / 255.0
                image_exp = np.expand_dims(image_np, axis=0)
                # Pass this image to classification model for prediction
                result = model.predict(image_exp)
                # Positive value predict face with mask, negative predicts face without mask
                # This binary classification + or - is learned purely by testing
                # However its fixed for one model
                if result > 0:
                    face_label.append("mask")
                    # print("Wearing Mask")
                else:
                    face_label.append("no mask")
                    # print("Not Wearing Mask")

        except Exception as e:
            print("Error in mask prediction ")
    # Count number of faces in a frame
    face_count_frame = len(faces)
    images_faces_count = len(images_faces)
    # just a validation that total number of faces should be equal or less than predicted ones
    if images_faces_count <= face_count_frame:
        for count in range(images_faces_count):
            startX = cordinates[count][0]
            startY =cordinates[count][1]
            endX =cordinates[count][2]
            endY =cordinates[count][3]
            face_lable = face_label[count]
            # Labelling faces on original frame based on their classification prediction
            if face_lable == "mask":
                cv2.putText(frame, "Wearing Mask", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,220,0),1)
                cv2.rectangle(frame,
                              (startX, startY), (endX, endY),
                              (0, 220, 0), 2)
            else:
                cv2.putText(frame, "No Mask", (startX, startY),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)
                cv2.rectangle(frame,
                              (startX, startY), (endX, endY),
                              (0, 0, 220), 2)
        frame = cv2.resize(frame, (640, 480))
        # Write the output video
        out.write(frame.astype('uint8'))
        cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break
# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
#This should do the job of blurring someone's face in an image, video or live feed
import os
import cv2
import mediapipe as mp        #Another library that must be installed (pip install mediapipe). Requires older version of numpy (before 2.0)
import argparse               #A library needed so you can choose between video or livestream
#pip install "numpy<2" must be done in terminal if numpy version is 2.0 or over. It changes it to 1.26.4

#Output directory

output_dir = './Outputs/'                   #Defines output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)                 #Creates the directory if it doesn't already exist

#Argparse thing so users can input images, videos or live feed

args = argparse.ArgumentParser()                    #Creates argument variable
args.add_argument("--mode", default='webcam')        #Allows user to input the mode they want (default is image)
args.add_argument("--filePath", default=None)       #ALlows user to input file path (defaults output folder)

args = args.parse_args()        #Don't know what this does


def process_img(img, face_detection):       #Creates a big section of code to be a function to recall (makes stuff more streamlined). This processes the image.
    H, W, _ =img.shape                  #Extracts height and width from image and stores it into variables H and W
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          #Converts image from BGR to RGB
    out = face_detection.process(img_rgb)                   #Created "out" object (the processed image) from the new face_detection object + rgb image

    #print(out.detections)                                  #Shows the detections from the object, showing where the face is (relative bounding boxes and confidence)
    
    if out.detections is not None:
        #Extracts information from the object in a better way than just spilling the beans
        for detection in out.detections:                        #For loop goes from 1 until the number of detections have completed. Usually only once.
            location_data = detection.location_data             #Extracts the object location_data from the object
            bbox = location_data.relative_bounding_box          #Extracts object for the bounding box from the location_data object

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height    #FINALLY unpacks these russian doll objects into useful variables for drawing

            x1 = int(x1*W)                                      #Because the bounding box is RELATIVE, they need to be converted to the actual image size
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            #BLURRING PART

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))  #Calculates blur in the specific bounding box of the face to the img, then applies it to the bounded box part of the img
    
    return img      #returns image



#Detect faces

mp_facedetection = mp.solutions.face_detection      #This runs mediapipe's "solutions.face_detection" algorithm and saves it as an object (mp_facedetection)

with mp_facedetection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:  #calls object, requires a minimum detection confidence of 50% and the 0 model (ideal for faces within 2m of camera. 1 model for within 5 metres), calls this variant of the object face_detection
    
    if args.mode in ['image']:
        #Read image
        img_path = args.filePath            #New place where the image will go
        img = cv2.imread(img_path)          #Reads where the image currently is
        img = process_img(img, face_detection)      #Calls function defined on line 13 to do all the stuff

        cv2.imshow('img', img)
        cv2.waitKey(0)

        #Save image

        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
    
    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)       #Takes video from the file path chosen
        ret, frame = cap.read()                     #Reads the capture and makes the frame variable

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame.shape[1], frame.shape[0])) #This .VideoWriter function not only has to save properly, but needs a bunch more parameters to work properly (30 is the fps)

        while ret:
            frame = process_img(frame, face_detection)      #Calls function again but replaces img with frame
            output_video.write(frame)                       #Writes frame onto video
            ret, frame = cap.read()                         #Reads the latest frame too
        
        cap.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)                           #capture is directly from webcam
        ret, frame = cap.read()

        while ret:
            ret, frame = cap.read()                         #Gets continuous ret and frame values from the video capture variable
            frame = process_img(frame, face_detection)

            cv2.imshow('CENSORED', frame)                   #Shows frame live
            if cv2.waitKey(1) & 0xFF == ord('p'):      #Shows images at the camera FPS and if someone presses the letter "p" then it will pause the webcam stream.
                break
        
        
        cap.release()
        cv2.destroyAllWindows()
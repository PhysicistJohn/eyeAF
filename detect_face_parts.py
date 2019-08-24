# import the necessary packages
from imutils import face_utils
import scipy
import imutils
import dlib
import cv2
import os


usingWebcam          = True
isMotionDetecting    = False
isFaceDetecting      = True


image_width          = 0
image_path           = "lena.bmp"

#This dat file is NOT MINE and NOT LICENSED for any use other than playing around personally!!
download_path        = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
zip_path             = download_path.split('/')[-1]
temp_path            = zip_path+'.download'
if not os.path.exists(shape_predictor_path) : 
    import shutil
    import requests
    import bz2
    
    if not os.path.exists(zip_path):
        with open(temp_path,'wb') as face_dat_file :
            face_dat_data = requests.get(download_path)
            face_dat_file.write(face_dat_data.content)
            shutil.move(temp_path,zip_path)
    
    with bz2.open(zip_path,'rb') as zip_file:
        data = zip_file.read()

    with open(shape_predictor_path,'wb') as dat_file:
        dat_file.write(data)

    #Cleanup zip file
    os.remove(zip_path)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


window_name = 'Face Detection'
window      = cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
cap         = cv2.VideoCapture(0)

if image_width : first_image = imutils.resize(cap.read()[1],width=image_width)
else           : first_image = cap.read()[1]
old_frame   = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
older_frame = old_frame.copy()

if not image_width : image_width = old_frame.shape[1]
eye_weight = 0.5
while True:
    # load the iscipyut image, resize it, and convert it to grayscale
    if usingWebcam:
        _,image = cap.read()
    else:
        image   = cv2.imread(image_path)


    image   = imutils.resize(image, width=image_width)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    if isMotionDetecting :
        kernel       = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        blur         = cv2.GaussianBlur( older_frame  ,(3,3),3)

        for i in range(3): blur  = cv2.GaussianBlur(blur,(3,3),3)

        #grad         = cv2.morphologyEx(blur,cv2.MORPH_GRADIENT,kernel)

        motion_image = ( blur - blur.min() ).astype(float)
        #motion_image-= motion_image.min()
        sub_image    = cv2.erode(motion_image,kernel,iterations=1).astype(float)

        sub_image   += gray.astype(float)
        sub_image    = (255 * sub_image/sub_image.max() ).astype(scipy.uint8)


        older_frame  = (older_frame + old_frame)/2
        old_frame    = (  old_frame + sub_image)/2

        sub_image    = cv2.applyColorMap( sub_image  ,cv2.COLORMAP_JET)

    else:
        sub_image = image

    rects = []
    if isFaceDetecting:
        # detect faces in the grayscale image
        rects        = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
        	# determine the facial landmarks for the face region, then
        	# convert the landmark (x, y)-coordinates to a NumPy array

            shape  = predictor(gray, rect)
            shape  = face_utils.shape_to_np(shape)
            output = sub_image # face_utils.visualize_facial_landmarks(sub_image, shape)

            #loop over the face parts individually
            eyes_location = []
            eyes_area     = []
            xs,ys         = [],[]
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                # extract the ROI of the face region as a separate image
                bbox = cv2.boundingRect(scipy.array([shape[i:j]]))
                (x, y, w, h) = bbox
                xs.append(x) ; xs.append(x+w)
                ys.append(y) ; ys.append(y+h)

                if 'eye' in name and not 'brow' in name:
                    eyes_location.append(bbox)
                    eyes_area.append(w*h)

            eye_idx      = scipy.argmax(eyes_area)
            eye_weight   = 0.25*eye_idx + 0.75*eye_weight
            eye_idx      = int(round(eye_weight))

            (x, y, w, h) = eyes_location[eye_idx]
            xr,yr        = int(w/2),int(h/2)
            xc,yc        = x+xr,y+yr
            cv2.rectangle(output,(x,y),(x + w , y + h),(0,255,0),3)
            cv2.ellipse(output,(xc,yc),(xr,yr),0,0,360,255,-1)

            eye_idx = not eye_idx
            (x, y, w, h) = eyes_location[eye_idx]
            xr,yr        = int(w/2),int(h/2)
            xc,yc        = x+xr,y+yr
            cv2.rectangle(output,(x,y),(x + w , y + h),(0,0,255),3)
            cv2.ellipse(output,(xc,yc),(xr,yr),0,0,360,255,-1)

            #Face rect
            minx,maxx = min(xs),max(xs)
            miny,maxy = min(ys),max(ys)
            miny      = int(max(0,miny - 0.2 * (maxy-miny)))
            cv2.rectangle(output,(minx,miny),(maxx ,maxy),(0,255,0),3)


            	# visualize all facial landmarks with a transparent overlay
            cv2.imshow(window_name, output)

    if not len(rects) or not isFaceDetecting : cv2.imshow(window_name, sub_image)




    keyPress = cv2.waitKey(1)



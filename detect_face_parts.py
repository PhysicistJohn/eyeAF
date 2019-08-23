# import the necessary packages
from imutils import face_utils
import scipy
import imutils
import dlib
import cv2


usingWebcam          = True
isMotionDetecting    = False
isFaceDetecting      = True


image_width          = 0
image_path           = "lena.bmp"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

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

    '''
    absgrad = scipy.absolute(gray)
    thresh = absgrad.mean() - absgrad.std()
    gray[absgrad<thresh] = thresh
    thresh = absgrad.mean() + absgrad.std()
    gray[absgrad>thresh] = thresh
    '''



    keyPress = cv2.waitKey(1)




'''
            #loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            		# clone the original image so we can draw on it, then
            		# display the name of the face part on the image
            		clone = image.copy()
            		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            			0.7, (0, 0, 255), 2)

            		# loop over the subset of facial landmarks, drawing the
            		# specific face part
            		for (x, y) in shape[i:j]:
            			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                # extract the ROI of the face region as a separate image
            		(x, y, w, h) = cv2.boundingRect(scipy.array([shape[i:j]]))
            		roi = image[y:y + h, x:x + w]
            		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            		# show the particular face part
            		cv2.imshow("ROI", roi)
            		cv2.imshow(window_name, clone)
            		cv2.waitKey(0)

'''
import cv2
import time


print('Program Starting.')
first_frame = None
video = cv2.VideoCapture(0)
# The below code will be made in order to prevent small light exposure when the camera 
# is firstly oppened, which may result on a wrong background if this is not done.  
check, frame = video.read()
time.sleep(5)

while True:
    status = 0
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 21 x 21 will be the height and width of the gaussian blurring filter, 
    # while 0 will be the value for the standard deviation.
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0) 
    if first_frame is None: # Add the background.
        first_frame = gray_frame
    diff_frame = cv2.absdiff(first_frame, gray_frame)
    threshold_diff_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_diff_frame = cv2.dilate(threshold_diff_frame, None, iterations=2)
    contours, _ = cv2.findContours(threshold_diff_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 30_000: # This value may be interpreted as the tolerance to noises. 
            continue
        status = 1 # Moving object was found.
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    if status == 1:
        cropped_image = frame[y:y+h, x:x+w]
        cv2.imshow('Cropped Image', cropped_image)
    else:
        cv2.imshow('Cropped Image', gray_frame)
    print(f'Status: {status}')
    cv2.imshow('Threshold Diff', threshold_diff_frame)
    cv2.imshow('Color Frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  
video.release()
cv2.destroyAllWindows

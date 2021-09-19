import cv2
from constants import *

def accumulated_average(frame, weights_for_background):

    global bg
    
    if bg is None:
        bg = frame.copy().astype("float")
        return None
    else:
        cv2.accumulateWeighted(frame, bg, weights_for_background)

def hand_isolate(frame, threshold=25):
    
    global bg
    
    subtracted = cv2.absdiff(bg.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
k = 0

while k != 27:

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    image_zone = frame[hand_zone_top:hand_zone_bottom, hand_zone_right:hand_zone_left]
    grayscale_frame = cv2.cvtColor(image_zone, cv2.COLOR_BGR2GRAY)
    grayscale_frame = cv2.GaussianBlur(grayscale_frame, (9, 9), 0)
    
    if frame_num < 60:
        accumulated_average(grayscale_frame, weights_for_background)
        if frame_num <= 59:
            cv2.putText(frame_copy, "Don't move, background is syncing", (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif frame_num <= 300: 
        hand = hand_isolate(grayscale_frame)
        cv2.putText(frame_copy, "Prepare hand gesture for " + str(element), (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
        if hand is not None:
            isolated, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (hand_zone_right, hand_zone_top)], -1, (255, 0, 0), 1)            
            cv2.putText(frame_copy, str(frame_num)+ " of 600 frames", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow("Isolated Hand Image", isolated)
            
    else: 
        hand = hand_isolate(grayscale_frame)
        if hand is not None:
            isolated, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (hand_zone_right, hand_zone_top)], -1, (255, 0, 0),1)
            cv2.putText(frame_copy, str(frame_num)+ " of 600 frames", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(frame_copy, str(images_taken) + " images taken " + "for " + str(element), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Isolated Hand Image", isolated)
            if images_taken <= 100:
                cv2.imwrite(test_data_path + "\\" + "0111_1" + "\\" + str(images_taken) + '.jpg', isolated)
            else:
                break
            images_taken +=1
        else:
            cv2.putText(frame_copy, "Nothing detected", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.rectangle(frame_copy, (hand_zone_left, hand_zone_top), (hand_zone_right, hand_zone_bottom), (57, 255, 20), 3)
    cv2.imshow("Sign Detection", frame_copy)
    frame_num += 1
    k = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
capture.release()
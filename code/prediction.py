import numpy as np
import cv2
import mediapipe as mp
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import statistics
from statistics import mode
from gtts import gTTS
from playsound import playsound
import os
from constants import *

model = keras.models.load_model(r"C:\Huzaifa\HR\asl\testbest_model_dataflair10.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

predictions = []
current_words = []
num_frames = 0
num_imgs_taken = 0




cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[hand_zone_top:hand_zone_bottom, hand_zone_right:hand_zone_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    # For webcam input:
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_copy.flags.writeable = False
        results = hands.process(frame_copy)

        # Draw the hand annotations on the image.
        frame_copy.flags.writeable = True
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_copy,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

    if num_frames < 70:
        
        accumulated_average(gray_frame, weights_for_background)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = accumulated_average(gray_frame)
        

        # Checking if we are able to detect the hand...
        if hand is not None:
            
            image, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (hand_zone_right, hand_zone_top)], -1, (255, 0, 0),1)
            
            cv2.imshow("Thesholded Hand Image", image)
            
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = np.reshape(image, (1,image.shape[0],image.shape[1],3))
            
            pred = model.predict(image)
            print(pred)
            if np.amax(pred) > 0.3:
                prediction = word_dict[np.argmax(pred)]
                
            else:
                prediction = None
            if prediction:
                predictions.append(word_dict[np.argmax(pred)])
            if len(predictions) >= 25 and prediction:
                recent_preds = predictions[-20:]
                counter = 0
                likely_pred = mode(recent_preds)
                for pred in recent_preds:
                    if pred == likely_pred:
                        counter += 1
                if counter >= 16 and likely_pred != previous_pred:
                   
                    current_words.append(likely_pred)

                    playsound("cached_tts\\" + likely_pred.lower() + ".mp3")

                    # os.remove(str(num_frames) + ".mp3")

                    if len(current_words) > 4:
                        current_words.pop(0)
                    previous_pred = likely_pred
            cv2.putText(frame_copy, str(current_words) , (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Draw ROI on frame_copy
    
    cv2.rectangle(frame_copy, (hand_zone_left, hand_zone_top), (hand_zone_right, hand_zone_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

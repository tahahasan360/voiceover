import cv2
word_list = ["0", "1", "2", "Hello", "World", "In", "We", "Leave", "No One", "Behind"]
word_dict = {}
for i in range(len(word_list)):
    word_dict[i] = word_list[i]

train_data_path = r'C:\Huzaifa\HR\asl\train'
test_data_path = r"C:\Huzaifa\HR\asl\test"

batch = len(word_list)

bg = None
weights_for_background = 0.5

hand_zone_top = 40
hand_zone_bottom = 450
hand_zone_right = 300
hand_zone_left = 600

frame_num = 0
images_taken = 0

#insert element being added to dataset 
element = str("Hello") 

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
import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # This object is used to detect hands

directory = './data' # Use data from new directory if you made your own base
data = []
labels = []
for class_dir in os.listdir(directory):
    for img_dir in os.listdir(os.path.join(directory,class_dir)):
        img = cv2.imread(os.path.join(directory,class_dir,img_dir))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_acc = []
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 1: # Only one hand can be shown as sign, e.g. the letter
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_acc.append(x)
                    data_acc.append(y)

            if len(data_acc) == 42: # Data will be passed only if all landmarks can be registered in image
                data.append(data_acc)
                labels.append(class_dir)

# print(data)
# print(labels)

with open("data.pickle", "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)

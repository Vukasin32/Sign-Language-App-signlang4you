import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # Objekat koji se koristi za detekciju

directory = './data' # Direktorijum u kom se nalaze slike
data = []
labels = []
for class_dir in os.listdir(directory):
    for img_dir in os.listdir(os.path.join(directory,class_dir)):
        img = cv2.imread(os.path.join(directory,class_dir,img_dir))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_acc = []
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 1: # Mora da bude detektovana jedna šaka
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_acc.append(x)
                    data_acc.append(y)

            if len(data_acc) == 42: # Podatak se smatra validnim samo ukoliku su sve značajne tačke šake detektovane
                data.append(data_acc)
                labels.append(class_dir)

# print(data)
# print(labels)

with open("data.pickle", "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)

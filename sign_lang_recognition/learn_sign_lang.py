import pickle
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

mp_hands = mp.solutions.hands # This object is used to detect hands
mp_drawing = mp.solutions.drawing_utils # This object is used to draw landmarks on image
mp_drawing_styles = mp.solutions.drawing_styles # This object is used to set style of landmarks

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# dict_letters represents which label corresponds to which letter in original images from data directory
dict_letters = {
    i + 1: letter
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Z'])
}

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_acc = []
    x_ = [] # List of landmarks x coord. of current prediction
    y_ = [] # List of landmarks y coord. of current prediction
    result = hands.process(frame_rgb)

    hand_label = None
    if result.multi_handedness:
        hand_label = result.multi_handedness[0].classification[0].label
        print(hand_label)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 1: # One hand regime
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_acc.append(x)
            data_acc.append(y)
            x_.append(x)
            y_.append(y)
        print(x_[8])
        print(y_[8])

        if hand_label == "Right":
            x_center = (min(x_) + max(x_)) / 2  # centralna vertikalna osa
            x_ = [2 * x_center - x for x in x_]

        if len(data_acc) == 42: # Prediction will be made only if all landmarks can be registered in image
            x1 = int(min(x_) * w) - 10
            x2 = int(max(x_) * w) + 10
            y1 = int(min(y_) * h) - 10
            y2 = int(max(y_) * h) + 10
            print(x1)
            data_acc = scaler.transform(np.array(data_acc).reshape(1, -1))

            label = model.predict(data_acc)
            prediction = dict_letters[int(label[0])]
            print(prediction)

            cv2.putText(frame, 'Predicted letter is: ' + prediction, (x1 - 15,y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 50, 205), 3,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 0), 3)

    elif result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2: # Two hands regime - Space regime
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        print('You are in Space regime (2 hands)')

        cv2.putText(frame, 'Space regime is activated!!!' , (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 150, 255), 3,
                    cv2.LINE_AA)

    cv2.imshow('APP WINDOW', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


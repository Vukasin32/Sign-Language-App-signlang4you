import os
import cv2

directory = './data' # Naziv direktorijuma gde će se čuvati slike

if not os.path.exists(directory):
    os.mkdir(directory)

num_of_classes = 22     # Broj simbola - broj klasa
size_of_classes = 800    # Broj primera svakog od simbola - broj primera svake klase

cap = cv2.VideoCapture(0)
for i in range(num_of_classes):
    if not os.path.exists(os.path.join(directory, str(i+1))):
        os.mkdir(os.path.join(directory, str(i+1)))

    while True:
        ret, frame = cap.read()

        cv2.putText(frame, f'Ready to collect class {i+1}?', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 100), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Use your right hand. For start ---> Press q', (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 100), 2,
                    cv2.LINE_AA)
        cv2.imshow('COLLECTING DATA', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyWindow('COLLECTING DATA')
            break

    print(f'Collecting images for class {i+1}')

    counter = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        cv2.imwrite(os.path.join(directory, str(i+1), str(counter) + '.jpg'), frame)
        counter += 1
        if counter == size_of_classes:
            break
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

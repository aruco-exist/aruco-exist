import os

import cv2
import cv2.aruco as aruco
import numpy as np

from info import info_dict

def detect_marker(frame, aruco_dict, parameters):
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    return corners, ids

def extract_main(corners, ids):
    main_idx = np.argmax([cv2.contourArea(c) for c in corners])
    return corners[main_idx], ids[main_idx][0]

def run_speech(info_dict, id):
    if id not in info_dict:
        return
    print(info_dict[id])
    os.system(f'espeak -v ko "{info_dict[id]}"')

def main():
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    print(parameters.maxMarkerPerimeterRate)

    prev_id = None

    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame, flipCode=1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids = detect_marker(gray, aruco_dict, parameters)
        if ids is not None: 
            corner, id = extract_main(corners, ids)
            print(id)
            if prev_id == id:
                continue
            if id is not None:
                print(id)
                prev_id = id
            run_speech(info_dict, id)

        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
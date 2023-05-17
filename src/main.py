import cv2
import cv2.aruco as aruco


def detect_marker(frame, aruco_dict, parameters):
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    return corners, ids


def main():
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame, flipCode=1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids = detect_marker(gray, aruco_dict, parameters)
        if ids is None: 
            break

        # frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
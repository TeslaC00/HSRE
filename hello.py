import cv2
import tensorflow as tf
from retinaface import RetinaFace


def main():
    print("Hello from hsre!")
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    image_path = "4.jpg"

    # Load image
    img = cv2.imread(image_path)

    # Detect faces
    resp = RetinaFace.detect_faces(image_path)
    print(resp)

    # Draw bounding boxes and landmarks
    for face_id, face_data in resp.items():
        face_area = list(map(int, face_data["facial_area"]))  # Convert to integers
        landmarks = face_data["landmarks"]

        # Draw bounding box
        x1, y1, x2, y2 = face_area
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks
        for key, point in landmarks.items():
            x, y = map(int, point)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(
                img,
                key,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # Show image
    cv2.imshow("Detected Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

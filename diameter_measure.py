import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(3.78, 3.78))
conversion_factor = 3.78 / 300


def create_gamma_table(gamma):
    invGamma = 1.0 / gamma
    table = np.power(np.arange(256) / 255.0, invGamma) * 255
    return table.astype("uint8")


gamma = 0.05
gamma_table = create_gamma_table(gamma)


def main():
    CHECKERBOARD = (6, 6)
    SQUARE_SIZE = 50
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    homography_matrix = None
    calibrated = False
    transformed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret_corners and not calibrated:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners_refined, ret_corners)
            cv2.putText(
                frame,
                "Checkerboard detected. Calibrating...",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Live Feed", frame)
            objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : CHECKERBOARD[1], 0 : CHECKERBOARD[0]].T.reshape(
                -1, 2
            )
            objp = objp * SQUARE_SIZE
            img_points = corners_refined.reshape(-1, 2)
            H, status = cv2.findHomography(img_points, objp[:, :2])

            if H is not None:
                homography_matrix = H
                calibrated = True
                print("Calibration successful. Applying perspective transformation.")
                cv2.putText(
                    frame,
                    "Calibration successful. Transforming...",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
            else:
                print("Error: Homography computation failed.")
                cv2.putText(
                    frame,
                    "Calibration failed. Try again.",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        elif calibrated and homography_matrix is not None:
            # Define the size of the transformed image
            transformed_width = CHECKERBOARD[1] * SQUARE_SIZE
            transformed_height = CHECKERBOARD[0] * SQUARE_SIZE

            # Apply perspective transformation
            transformed_frame = cv2.warpPerspective(
                frame, homography_matrix, (transformed_width, transformed_height)
            )
            corrected_frame = cv2.LUT(transformed_frame, gamma_table)
            gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            cv2.imshow("Thresholded Image", thresholded)

            bright_pixels = np.where(thresholded == 255)
            x_coords = bright_pixels[1]  # x positions
            y_coords = bright_pixels[0]  # y positions
            ax.clear()
            ax.plot(x_coords * conversion_factor, y_coords * conversion_factor, "ro")
            ax.set_xlim((0.0, 3.78))
            ax.set_ylim((0.0, 3.78))
            plt.draw()
            plt.pause(0.01)

            # cv2.imshow("Live Feed", frame)

        else:
            # Display instructions on the frame
            cv2.putText(
                frame,
                "Present a checkerboard to the camera.",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting program.")
            break
        elif key == ord("r"):
            if calibrated:
                print("Resetting calibration.")
                homography_matrix = None
                calibrated = False
                cv2.destroyWindow("Transformed Feed")
            else:
                print("Calibration not yet performed.")

    # Release resources
    plt.ioff()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

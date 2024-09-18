import cv2
import numpy as np


def process_frame(frame):                                                   # Function for processing frame

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                    # Turn frame into grayscale for further comparasons

    norm_image = cv2.normalize(gray_image, None, 0, 255,cv2.NORM_MINMAX)    # Normalize the image for better heatmap contrast

    inverted_image = 255 - norm_image                                       # Switch between cold and hot highlighted ( Heat and cold area are switched, so I invert it )

    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)             # Apply Gaussian Blur to reduce noises

    heatmap = cv2.applyColorMap(blurred_image, cv2.COLORMAP_JET)            # Create a heatmap

    return heatmap


def main():                                                                 # Main function

    cap = cv2.VideoCapture(0)                                               # Open webcam on 0 port

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        heatmap = process_frame(frame)                                       # Process the current frame

        cv2.imshow('Original Frame', frame)                                  # Show original frame
        cv2.imshow('Heat Map', heatmap)                                      # Show thermal frame

        if cv2.waitKey(1) & 0xFF == ord('q'):                                # Press q to stop
            break

    cap.release()
    cv2.destroyAllWindows()                                                  # Close all windows

if __name__ == "__main__":                                                   # Main
    main()
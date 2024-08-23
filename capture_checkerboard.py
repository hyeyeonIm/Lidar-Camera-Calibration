#!/usr/bin/env python
# -- coding: utf-8 --
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import datetime

# Initialize CvBridge
CV_BRIDGE = CvBridge()

if __name__ == '__main__':
    pub = rospy.Publisher('image_topic', Image, queue_size=1)
    rospy.init_node('image_publisher')

    # Set up the directory for saving images
    save_dir = os.path.expanduser('~/catkin_ws/src/cali/checkerboard')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video file
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Display the annotated frame
            cv2.imshow("image", frame)
            msg = CV_BRIDGE.cv2_to_imgmsg(frame, 'bgr8')

            # Save the frame if 's' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Get the current time and format it as a string
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"checkerboard_{timestamp}.jpg"
                filepath = os.path.join(save_dir, filename)
                # Save the image
                cv2.imwrite(filepath, frame)
                print(f"Image saved: {filepath}")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

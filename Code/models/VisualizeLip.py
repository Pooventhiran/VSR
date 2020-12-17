# Copyright {2017} {Amirsina Torfi}
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# The code has minor modifications accordingly for our work

import numpy as np
import cv2
import dlib
import argparse
import os
import skvideo.io


"""
PART1: Construct the argument parse and parse the arguments
"""

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video file")
ap.add_argument("-f", "--fps", type=int, default=10, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
args = vars(ap.parse_args())
args["output"] = os.path.join(os.path.dirname("VisualizeLip.py"), "..", "data")

for spk in os.listdir(args["input"]):
    for word in range(1, 11):
        for i in range(1, 11):

            """
            PART2: Calling and defining required parameters for:

                   1 - Processing video for extracting each frame.
                   2 - Lip extraction from frames.
            """

            # Dlib requirements.
            predictor_path = "./shape_predictor_68_face_landmarks.dat"
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            mouth_destination_path = os.path.join(
                args["output"], spk, "words", "%02d" % word, "%02d" % i
            )
            if not os.path.exists(mouth_destination_path):
                os.makedirs(mouth_destination_path)

            inputparameters = {}
            outputparameters = {}
            reader = skvideo.io.FFmpegReader(
                os.path.join(
                    args["input"],
                    spk,
                    "words",
                    "%02d" % word,
                    "%02d" % i,
                    "word%02dutterance%02d.mp4" % (word, i),
                ),
                inputdict=inputparameters,
                outputdict=outputparameters,
            )
            video_shape = reader.getShape()
            (num_frames, h, w, c) = video_shape
            print(num_frames, h, w, c)

            activation = []
            max_counter = 300
            total_num_frames = int(video_shape[0])
            num_frames = min(total_num_frames, max_counter)
            counter = 0
            font = cv2.FONT_HERSHEY_SIMPLEX

            writer = skvideo.io.FFmpegWriter(
                os.path.join(
                    args["output"],
                    spk,
                    "words",
                    "%02d" % word,
                    "%02d" % i,
                    "lip-word%02dutterance%02d.mp4" % (word, i),
                )
            )

            width_crop_max = 0
            height_crop_max = 0

            for frame in reader.nextFrame():
                print("frame_shape:", frame.shape)

                if counter > num_frames:
                    break

                detections = detector(frame, 1)

                marks = np.zeros((2, 20))

                Features_Abnormal = np.zeros((190, 1))

                print(len(detections))
                if len(detections) > 0:
                    for k, d in enumerate(detections):
                        print("Location:", d)
                        shape = predictor(frame, d)

                        co = 0
                        for ii in range(48, 68):
                            X = shape.part(ii)
                            A = (X.x, X.y)
                            marks[0, co] = X.x
                            marks[1, co] = X.y
                            co += 1

                        X_left, Y_left, X_right, Y_right = [
                            int(np.amin(marks, axis=1)[0]),
                            int(np.amin(marks, axis=1)[1]),
                            int(np.amax(marks, axis=1)[0]),
                            int(np.amax(marks, axis=1)[1]),
                        ]

                        X_center = (X_left + X_right) / 2.0
                        Y_center = (Y_left + Y_right) / 2.0

                        border = 10
                        X_left_new = X_left - border
                        Y_left_new = Y_left - border
                        X_right_new = X_right + border
                        Y_right_new = Y_right + border

                        # Width and height for cropping(before and after considering the border).
                        width_new = X_right_new - X_left_new
                        height_new = Y_right_new - Y_left_new
                        width_current = X_right - X_left
                        height_current = Y_right - Y_left

                        # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
                        if width_crop_max == 0 and height_crop_max == 0:
                            width_crop_max = width_new
                            height_crop_max = height_new
                        else:
                            width_crop_max += 1.5 * np.maximum(
                                width_current - width_crop_max, 0
                            )
                            height_crop_max += 1.5 * np.maximum(
                                height_current - height_crop_max, 0
                            )

                        X_left_crop = int(X_center - width_crop_max / 2.0)
                        X_right_crop = int(X_center + width_crop_max / 2.0)
                        Y_left_crop = int(Y_center - height_crop_max / 2.0)
                        Y_right_crop = int(Y_center + height_crop_max / 2.0)

                        if (
                            X_left_crop >= 0
                            and Y_left_crop >= 0
                            and X_right_crop < w
                            and Y_right_crop < h
                        ):
                            mouth = frame[
                                Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :
                            ]

                            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                            cv2.imwrite(
                                os.path.join(
                                    mouth_destination_path,
                                    "color_" + "_" + "%03d.jpg" % (counter + 1),
                                ),
                                mouth_gray,
                            )

                            print("The cropped mouth is detected ...")
                            activation.append(1)
                        else:
                            cv2.putText(
                                frame,
                                "The full mouth is not detectable. ",
                                (30, 30),
                                font,
                                1,
                                (0, 255, 255),
                                2,
                            )
                            print("The full mouth is not detectable. ...")
                            activation.append(0)

                else:
                    cv2.putText(
                        frame,
                        "Mouth is not detectable. ",
                        (30, 30),
                        font,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    print("Mouth is not detectable. ...")
                    activation.append(0)

                if activation[counter] == 1:
                    cv2.rectangle(
                        frame,
                        (X_left_crop, Y_left_crop),
                        (X_right_crop, Y_right_crop),
                        (0, 255, 0),
                        2,
                    )

                print("frame number %d of %d" % (counter, num_frames))

                print(
                    "writing frame %d with activation %d"
                    % (counter + 1, activation[counter])
                )
                writer.writeFrame(frame)
                counter += 1

            writer.close()

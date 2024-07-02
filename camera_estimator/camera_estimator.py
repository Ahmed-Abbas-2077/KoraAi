import pickle
import cv2
import numpy as np
import os


class CameraEstimator:
    def __init__(self, frame):

        self.minimum_distance = 5

        # Set the parameters for the Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Convert the frame to gray scale
        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask of zeros with the same shape as the frame
        mask_features = np.zeros_like(first_frame_gray)

        # Set the first rectangle of the mask to 1
        mask_features[:, 0:20] = 1

        # Set the last rectangle of the mask to 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    position = track["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    positions_adjusted = (
                        position[0] + camera_movement[0], position[1] + camera_movement[1])

                    tracks[object][frame_num][track_id]["position_adjusted"] = positions_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calculate the euclidean distance between the new and old features
                distance = ((new_features_point[0] - old_features_point[0]) ** 2 + (
                    new_features_point[1] - old_features_point[1]) ** 2) ** 0.5

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = old_features_point[0] - \
                        new_features_point[0]
                    camera_movement_y = old_features_point[1] - \
                        new_features_point[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [
                    camera_movement_x, camera_movement_y]

                old_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]

            # Draw the camera movement on the frame
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames

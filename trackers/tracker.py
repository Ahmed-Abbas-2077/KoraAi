from ultralytics import YOLO
import supervision as sv
import pandas as pd
import numpy as np
import pickle
import cv2
import os


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_positions_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    bbox = track["bbox"]
                    if object == "ball":
                        # position is the center of the ball
                        position = [(bbox[0] + bbox[2]) // 2,
                                    (bbox[1] + bbox[3]) // 2]
                    else:
                        # position is the bottom center of the player
                        position = [(bbox[0] + bbox[2]) // 2, bbox[3]]

                    # make sure position is an integer
                    position[0] = int(position[0])
                    position[1] = int(position[1])

                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolating missing values in the DataFrame.
        df_ball_positions = df_ball_positions.interpolate()

        # Backward fill to fill the first few missing values.
        df_ball_positions = df_ball_positions.bfill()

        # Converting the processed DataFrame back into the original nested dictionary structure.
        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(
                frames[i:i + batch_size], conf=0.1)
            detections += detection_batch

        return detections

    def get_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            print(cls_names)

            # convert detections to format required by tracker (supervision)
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeepers to players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Track objects

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # save tracks for players and referees
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # save tracks for ball
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) // 2
        radius_x = (x2 - x1) // 2
        radius_y = (y2 - y1) // 2

        # @_typing.overload
        # def ellipse(img: cv2.typing.MatLike, center: cv2.typing.Point, axes: cv2.typing.Size,
        #             angle: float, startAngle: float, endAngle: float, color: cv2.typing.Scalar, thickness: int = ...,
        #             lineType: int = ..., shift: int = ...) -> cv2.typing.MatLike: ...
        # parameters: image, center, axes, angle, startAngle, endAngle, color, thickness
        frame = cv2.ellipse(frame, (int(x_center), int(y2)),
                            (int(1.5*radius_x), int(0.85*radius_x)), 0, -45, 235, color, 2, cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            # draw track id
            x1_text = x1_rect + 12
            if track_id > 99:  # 3 digit track id
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        xc = int((bbox[0] + bbox[2]) / 2)

        # draw triangle
        triangle_points = np.array([
            [xc, y],
            [xc-10, y-20],
            [xc+10, y-20],
        ])

        # draw filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

        # draw triangle outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_now = team_ball_control[:frame_num+1]

        # Get the number of times each team had ball control
        team_1_num_frames = team_ball_control_till_now[team_ball_control_till_now == 1].shape[0]
        team_2_num_frames = team_ball_control_till_now[team_ball_control_till_now == 2].shape[0]

        # Get the percentage of time each team had ball control
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1_percentage = 0
            team_2_percentage = 0
        else:
            team_1_percentage = team_1_num_frames / total_frames * 100
            team_2_percentage = team_2_num_frames / total_frames * 100

        cv2.putText(frame, f"Team 1: {team_1_percentage:.2f}%",
                    (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        cv2.putText(frame, f"Team 2: {team_2_percentage:.2f}%",
                    (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []

        # draw ellipses on the frames
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], (0, 0, 255))

            # draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"], (0, 255, 255))

            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(
                    frame, ball["bbox"], (0, 255, 0))

            # draw team ball
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control)

            output_frames.append(frame)

        return output_frames

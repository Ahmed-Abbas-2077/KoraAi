import cv2


class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.fps = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referee":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                # this is to ensure that the last frame is not out of bounds
                last_frame = min(
                    frame_num + self.frame_window, number_of_frames-1)

                for track_id, track in object_tracks[frame_num].items():
                    # if the track is not in the last frame, skip it
                    # this to make sure that we have a track for the object in the last frame
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    # calculate euclidean distance
                    distance_covered = ((start_position[0] - end_position[0]) ** 2 + (
                        start_position[1] - end_position[1]) ** 2) ** 0.5

                    # calculate time taken
                    time_taken = (last_frame - frame_num) / self.fps

                    # calculate speed (meters per second)
                    speed_ms = distance_covered / time_taken
                    speed_kmh = speed_ms * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]["speed"] = speed_kmh
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance[object][track_id]

    def draw_speed_and_distance(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referee":
                    continue

                for track_id, track in object_tracks[frame_num].items():

                    if "speed" in track:
                        speed = track.get("speed", None)
                        distance = track.get("distance", None)
                        if speed is None or distance is None:
                            continue

                        bbox = track["bbox"]
                        # foot position [bbox[0] + bbox[2] // 2, bbox[3]]
                        position = [bbox[2], bbox[3]]
                        # move the text down. This is to avoid overlapping with the bbox
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames

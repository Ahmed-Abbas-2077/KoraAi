import numpy as np
import cv2


class ViewTransformer:
    def __init__(self):
        # Define the court dimensions
        court_width = 68
        court_length = 23.32

        # pixel vertices are the perceived vertices of the court in the image
        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        # target vertices are actual court dimensions
        self.target_vertices = np.array([[0, court_width],
                                         [0, 0],
                                         [court_length, 0],
                                         [court_length, court_width]])

        # float the vertices
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # prespective transform is the transformation matrix
        self.prespective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = int(point[0]), int(point[1])  # point to be transformed
        # check if point is inside the polygon
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # reshape the point to 3D
        reshaped_point = point.reshape(1, -1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.prespective_transform)

        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    position = track["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze(
                        ).tolist()  # remove extra dimensions

                    # add the transformed position to the tracks
                    tracks[object][frame_num][track_id]["position_transformed"] = position_transformed

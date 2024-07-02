from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_estimator import CameraEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    # read video
    # path to the video file. replace this with the path to the video file you want to process
    video_frames = read_video(
        'D:/Computer_Vision/Kora.ai/Kora.ai/input_videos/08fd33_4.mp4')

    # intialize tracker
    tracker = Tracker('D:/Computer_Vision/Kora.ai/Kora.ai/models/best.pt')

    tracks = tracker.get_tracks(video_frames, read_from_stub=True,
                                stub_path='D:/Computer_Vision/Kora.ai/Kora.ai/stubs/track_stubs.pkl')

    # get object positions
    tracker.add_positions_to_tracks(tracks)

    # camera movement estimation
    camera_estimator = CameraEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='D:/Computer_Vision/Kora.ai/Kora.ai/stubs/camera_movement_stubs.pkl')

    camera_estimator.adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_tracks in enumerate(tracks["players"]):
        for player_id, player_track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 player_track["bbox"],
                                                 player_id)

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_tracks in enumerate(tracks["players"]):
        # get ball bbox
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]

        # assign ball to player
        assigned_player = player_assigner.assign_ball_to_player(
            player_tracks, ball_bbox)

        if assigned_player != -1:
            # update tracks
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            tracks["ball"][frame_num][1]["assigned_player"] = assigned_player

            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"])
        else:
            # if the ball is not assigned to any player, assign it to the previous team that had it
            # Otherwise, assign it to None
            team_ball_control.append(
                team_ball_control[-1] if len(team_ball_control) else None)

    team_ball_control = np.array(team_ball_control)

    # draw output
    # draw objects tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    # draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    # save the video
    # path to save the output video. replace this with the path where you want to save the output video
    save_video(
        output_video_frames, 'D:/Computer_Vision/Kora.ai/Kora.ai/output_videos/output.avi')


if __name__ == '__main__':
    main()

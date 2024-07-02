import sys


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        center = (ball_bbox[0] + ball_bbox[2]) / \
            2, (ball_bbox[1] + ball_bbox[3]) / 2
        ball_position = center

        minimum_distance = 999999
        assigned_player = -1

        for player_id, player in players.items():
            player_position = player["bbox"]

            # get euclidean distance between left corner (left leg) of player and center of the ball
            distance_left = ((player_position[0] - ball_position[0]) ** 2 + (
                player_position[3] - ball_position[1]) ** 2) ** 0.5

            # same for right corner (right leg) of player
            distance_right = ((player_position[2] - ball_position[0]) ** 2 + (
                player_position[3] - ball_position[1]) ** 2) ** 0.5

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

import cv2
import numpy as np


class VideoRecorder(object):
    # Courtesy: https://github.com/vdean/audio-curiosity
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def add_metric_to_frame(self, image, metric, t, position):
        rounded_metric = str(round(metric[t], 1))
        normalized_metric = metric[t] / np.max(metric)

        text_color = (0, 255, 255 * (1.0-normalized_metric))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, rounded_metric, position, font, 0.4, text_color, 1, cv2.LINE_AA)

    def save_video(self, n_updates, observations, internal_rewards, name=""):
        video_path = self.log_dir + '/' + str(n_updates) + name + '.mp4'
        frame_rate = 15

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (84, 84))
        for t in range(observations.shape[1]):
            env = 0  # Only record video for environment 0
            image = observations[env, t, :, :, 0]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            self.add_metric_to_frame(image, internal_rewards[env], t, position=(2,14))

            video.write(image)
        video.release()

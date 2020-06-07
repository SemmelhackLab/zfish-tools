import cv2

from zfishtools import DisplayWithTrackBars, Video
from .image_processing import draw_body_parts


class ManualEyeTracker(DisplayWithTrackBars):
    def __init__(self, video_path, rois=None, image_kwargs=None):
        if image_kwargs is None:
            image_kwargs = {}
        video = Video(video_path)
        if rois is None:
            rois = cv2.selectROIs("Select ROIs", video[0], showCrosshair=False)
            cv2.destroyAllWindows()

        image_kwargs['rois'] = rois

        super().__init__(video, ['radius', 'threshold'], [(1, 100, 50), (-50, 0, 50)],
                         image_func=draw_body_parts, image_kwargs=image_kwargs)

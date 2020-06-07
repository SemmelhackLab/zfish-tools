import cv2


class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.__frames = Video.load_frames(video_path)

    def __len__(self):
        return len(self.__frames)

    def __getitem__(self, item):
        index = int(item)
        return self.__frames[index if index < len(self) else len(self) - 1]

    @staticmethod
    def load_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        # frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        #           for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
        frames = [cap.read()[1] for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
        cap.release()
        return frames

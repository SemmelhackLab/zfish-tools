import cv2


class Display:
    def __init__(self, video, frame_number=0, image_func=None, image_kwargs=None):
        self.video = video
        self.current_raw_frame = video[0].copy()
        self.current_frame= self.current_raw_frame.copy()
        self.window = video.video_path
        self.frame_number = frame_number
        self.image_func = image_func if image_func else lambda *args: args[0]
        self.image_kwargs = image_kwargs if image_kwargs else {}
        cv2.namedWindow(self.window)
        cv2.createTrackbar('Frame', self.window, 0, len(self.video), self.frame_bar_update)
        cv2.setTrackbarPos('Frame', self.window, frame_number)

        self.show()

    def update_current_frame(self):
        self.current_frame = self.image_func(self.current_raw_frame, **self.image_kwargs)

    def show(self, image=None):
        if image:
            cv2.imshow(self.window, image)
        else:
            cv2.imshow(self.window, self.current_frame)

    def frame_bar_update(self, frame_number):
        self.frame_number = frame_number
        self.current_raw_frame = self.video[self.frame_number].copy()
        self.update_current_frame()
        self.show()

    def close(self):
        cv2.destroyWindow(self.window)


class DisplayWithTrackBars(Display):
    def __init__(self, video, names=(), values=(), frame_number=0, image_func=None, image_kwargs=None):
        if image_kwargs is None:
            image_kwargs = {}

        for name, value in zip(names, values):
            image_kwargs[name] = value[2]

        super().__init__(video, frame_number, image_func, image_kwargs)

        for name, value in zip(names, values):
            cv2.createTrackbar(name, self.window, value[0], value[1], self.create_track_bar_callback(name))
            cv2.setTrackbarPos(name, self.window, value[2])

    def create_track_bar_callback(self, track_bar_name):
        return lambda value: self.update_param(track_bar_name, value)

    def update_param(self, param, value):
        self.image_kwargs[param] = value
        self.update_current_frame()
        self.show()

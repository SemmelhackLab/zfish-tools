import cv2
import pandas as pd
import yaml
import os
import numpy as np
from pathlib import Path
from tkinter import filedialog
from tkinter import Tk

from zfishtools import ManualEyeTracker, get_tail_points


class DLCLabeler(ManualEyeTracker):
    brush_color = (128, 0, 128)
    brush_radius = 3
    n_points = 51

    def __init__(self, video_path=None, config_path=None, rois=None):
        if video_path is None:
            root = Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(title="Select video", filetypes=[("avi", "*.avi")])
            root.destroy()

        if config_path is None:
            root = Tk()
            root.withdraw()
            config_path = filedialog.askopenfilename(title="Select DeepLabCut config", filetypes=[("yaml", "*.yaml")])
            root.destroy()

        self.config_path = config_path
        with open(config_path) as file:
            self.config = yaml.safe_load(file)

        super().__init__(video_path, rois, image_kwargs={'display': self})

        self.scorer = self.config['scorer']
        self.dir = os.path.split(self.config_path)[0] + '/labeled-data/' + self.video_name
        self.dataframe_path = os.path.join(self.dir, "CollectedData_" + self.scorer + '.h5')

        self.columns = pd.MultiIndex.from_product([[self.config['scorer']], self.config['bodyparts'], ['x', 'y']],
                                                  names=['scorer', 'bodyparts', 'coords'])
        if os.path.isfile(self.dataframe_path):
            self.dataframe = pd.read_hdf(self.dataframe_path, 'df_with_missing')
        else:
            self.dataframe = pd.DataFrame([], columns=self.columns)

        self.captured_frames = {}
        self.draw_mode = False
        self.tail_frame = self.current_raw_frame.copy()
        cv2.setMouseCallback(self.window, self.draw)
        print("Label tail:")
        print("1. Draw tail on a selected frame")
        print("2. Press enter to save labels")
        print()
        print("Press ESC when finish labeling all the frames needed")

        while True:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                break
            k = cv2.waitKey(100)
            if k == 13:
                self.capture()
            elif k == 27:
                cv2.destroyAllWindows()
                self.save()
                break

    def frame_bar_update(self, frame_number):
        self.tail_frame = self.video[frame_number].copy()
        self.image_kwargs['tail_points'] = []
        super().frame_bar_update(frame_number)

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_mode = True
            self.tail_frame = self.current_raw_frame.copy()
            self.show()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw_mode:
                cv2.circle(self.current_frame, (x, y), self.brush_radius, self.brush_color, -1)
                cv2.circle(self.tail_frame, (x, y), self.brush_radius, self.brush_color, -1)
            self.show()
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw_mode = False
            self.image_kwargs['tail_points'] = get_tail_points(self.tail_frame, self.brush_color, self.n_points)
            self.update_current_frame()

    @property
    def video_name(self):
        return os.path.splitext(os.path.split(self.video.video_path)[-1])[0]

    def format_current_frame_number(self):
        return 'labeled-data\\' + self.video_name + '\\img' + (
                    "{:0" + str(len(str(len(self.video)))) + "d}").format(self.frame_number) + '.png'

    def capture(self, *args):
        try:
            tail_points = self.image_kwargs['tail_points']
            print(f"Frame {self.frame_number} captured.")
        except KeyError:
            print("Error: please draw tail before capturing frame")
            return

        eye_points = self.image_kwargs['eye_points']

        self.captured_frames[self.frame_number] = self.image_kwargs.copy()
        all_points = np.concatenate([np.array(eye_points).ravel(), tail_points.ravel()]).astype(int)
        if len(all_points) == self.n_points * 2 + 18:
            self.dataframe.loc[self.format_current_frame_number()] = all_points

    def save(self):
        directory = os.path.split(self.config_path)[0] + '/labeled-data/' + self.video_name
        Path(directory).mkdir(parents=True, exist_ok=True)
        for i in self.dataframe.index:
            frame_num = int(i.split('img')[-1].split('.')[0])
            cv2.imwrite(os.path.join(directory, os.path.split(i)[-1]), self.video[frame_num])

        self.dataframe = self.dataframe.sort_index()
        self.dataframe.astype(int).to_hdf(os.path.join(self.dir, "CollectedData_" + self.scorer + '.h5'),
                                          'df_with_missing', format='table', mode='w')

        r3, r1 = self.video[0].shape[:2]

        self.config['video_sets'][self.video.video_path] = {'crop': str([0, r1, 0, r3])[1:-1]}

        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

        print(f"Labels of {self.video_name} is saved to {self.dir}")

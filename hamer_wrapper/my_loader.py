from .utils import *


class MyLoader:
    def __init__(self, sequence_folder):
        self._data_folder = Path(sequence_folder)
        self._load_meta_data()
        self._load_hand_info()

    def _load_meta_data(self):
        metadata_file = self._data_folder / "meta.json"
        metadata = read_data_from_json(metadata_file)
        self._rs_serials = metadata["realsense"]["serials"]
        self._rs_width = metadata["realsense"]["width"]
        self._rs_height = metadata["realsense"]["height"]
        self._num_frames = metadata["num_frames"]

    def _load_hand_info(self):
        if not (self._data_folder / "hand_benchmark.json").exists():
            self._hand_info = None
            return
        with open(self._data_folder / "hand_benchmark.json", "r") as file:
            data = json.load(file)
        self._hand_info = data

    def get_cv_image(self, serial, frame_id):
        img_file = self._data_folder / serial / f"color_{frame_id:06d}.jpg"
        img = cv2.imread(str(img_file))
        return img

    def _get_valid_marks_num(self, marks):
        count = 0
        for mark in marks:
            if np.all(mark != -1):
                count += 1
        return count

    def get_data_item(self, serial, frame_id):
        right_flags = []
        boxes = []
        img_cv2 = cv2.imread(
            str(self._data_folder / serial / f"color_{frame_id:06d}.jpg")
        )
        if self._hand_info is None:
            return img_cv2, boxes, right_flags
        hand_info = self._hand_info[serial][str(frame_id)]
        for idx, side in enumerate(["left", "right"]):
            mks = np.array(hand_info["landmarks_2d"][side])
            valid_mks_num = self._get_valid_marks_num(mks)
            if valid_mks_num < 3:
                continue
            boxes.append(np.array(hand_info["bbox"][side]))
            right_flags.append(idx)
        boxes = np.array(boxes)
        right_flags = np.array(right_flags)
        return img_cv2, boxes, right_flags

from .utils import *
from .my_loader import MyLoader

add_path(PROJ_ROOT)
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from .keypoints_detector import KeyPointDetector


class HamerWrapper:
    def __init__(self, use_detector=False, device="cuda"):
        self._device = device
        self._use_detector = use_detector
        self._model, self._model_cfg = self._load_hamer_model()

        if use_detector:
            self._kpt_detector = KeyPointDetector(device)

    def _load_hamer_model(self):
        model, model_cfg = load_hamer(
            PROJ_ROOT / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
        )
        model = model.to(self._device)
        model.eval()
        return model, model_cfg

    def _get_bbox_and_right_flag_from_vitposes(
        self, vitposes_out, kpt_score_thresh=0.55, min_kpt_num=3
    ):
        bboxes = []
        is_right = []

        for vitposes in vitposes_out:
            right_hand_kpts = vitposes["keypoints"][-21:]
            left_hand_kpts = vitposes["keypoints"][-42:-21]

            valid_mask = right_hand_kpts[:, 2] > kpt_score_thresh
            if valid_mask.sum() > min_kpt_num:
                bbox = [
                    right_hand_kpts[valid_mask, 0].min(),
                    right_hand_kpts[valid_mask, 1].min(),
                    right_hand_kpts[valid_mask, 0].max(),
                    right_hand_kpts[valid_mask, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(1)

            valid_mask = left_hand_kpts[:, 2] > kpt_score_thresh
            if valid_mask.sum() > min_kpt_num:
                bbox = [
                    left_hand_kpts[valid_mask, 0].min(),
                    left_hand_kpts[valid_mask, 1].min(),
                    left_hand_kpts[valid_mask, 0].max(),
                    left_hand_kpts[valid_mask, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(0)
        bboxes = np.array(bboxes)
        is_right = np.array(is_right)

        return bboxes, is_right

    def predict_with_vitpose(self, img_cv2):
        h, w = img_cv2.shape[:2]
        pred_keypoints_2d_full = np.full((2, 21, 2), -1, dtype=int)
        # Predict keypoints using keypoint detector
        vitposes_out = self._kpt_detector.predict_vitposes(img_cv2)
        bboxes, is_right = self._get_bbox_and_right_flag_from_vitposes(vitposes_out)

        if len(bboxes) == 0:
            return pred_keypoints_2d_full

        boxes = np.stack(bboxes)
        is_right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(
            self._model_cfg, img_cv2, boxes, is_right, rescale_factor=2.0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )

        for batch in dataloader:
            batch = recursive_to(batch, self._device)
            with torch.no_grad():
                out = self._model(batch)

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            pred_keypoints_2d = out["pred_keypoints_2d"]

            for kpts, box_c, box_s, right_flag in zip(
                pred_keypoints_2d, box_center, box_size, is_right
            ):
                if right_flag == 0:  # left hand
                    kpts[:, 0] *= -1
                kpts = kpts * box_s + box_c
                kpts = kpts.cpu().numpy().astype(int)
                msk = (
                    (kpts[:, 0] >= 0)
                    & (kpts[:, 0] < w)
                    & (kpts[:, 1] >= 0)
                    & (kpts[:, 1] < h)
                )
                kpts[~msk] = -1
                if right_flag == 0:
                    pred_keypoints_2d_full[1] = kpts
                elif right_flag == 1:
                    pred_keypoints_2d_full[0] = kpts

        return pred_keypoints_2d_full

    def predict(self, img_cv2, boxes=None, right_flags=None):
        h, w = img_cv2.shape[:2]
        pred_keypoints_2d_full = np.full((2, 21, 2), -1, dtype=int)

        if self._use_detector:
            vitposes_out = self._kpt_detector.predict_vitposes(img_cv2)
            boxes, right_flags = self._get_bbox_and_right_flag_from_vitposes(
                vitposes_out
            )

        if boxes is None or len(boxes) == 0:
            return pred_keypoints_2d_full
        
        

        dataset = ViTDetDataset(
            self._model_cfg, img_cv2, boxes, right_flags, rescale_factor=2.0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        for batch in dataloader:
            batch = recursive_to(batch, self._device)
            with torch.no_grad():
                out = self._model(batch)

            box_c = batch["box_center"].float()
            box_s = batch["box_size"].float()
            kpts = out["pred_keypoints_2d"][0]
            is_right = batch["right"].item()

            if is_right == 0:
                kpts[:, 0] *= -1
            kpts = kpts * box_s + box_c
            kpts = kpts.cpu().numpy().astype(int)
            msk = (
                (kpts[:, 0] >= 0)
                & (kpts[:, 0] <= w - 1)
                & (kpts[:, 1] >= 0)
                & (kpts[:, 1] <= h - 1)
            )
            kpts[~msk] = -1
            if is_right == 0:
                pred_keypoints_2d_full[1] = kpts
            elif is_right == 1:
                pred_keypoints_2d_full[0] = kpts
            else:
                raise ValueError(f"Invalid right flag: {is_right}")
        return pred_keypoints_2d_full

from .utils import *

add_path(PROJ_ROOT)

from vitpose_model import ViTPoseModel
from detectron2.config import LazyConfig
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

class KeyPointDetector:
    def __init__(self, device="cuda") -> None:
        self._device = device
        self._detector = self._load_body_detector()
        self._cpm = ViTPoseModel(device)

    def _load_body_detector(self):
        cfg_path = PROJ_ROOT / "hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(1):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector
    
    def _detect_humans(self, img_cv2):
        det_out = self._detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        detect_results = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        return detect_results

    def predict_vitposes(self, img_cv2):
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # Detect humans in image
        detect_results  = self._detect_humans(img_cv2)

        # Detect pose keypoints for each person
        vitposes_out = self._cpm.predict_pose(
            image=img_rgb,
            det_results=detect_results,
            box_score_threshold=0.5,
        )

        return vitposes_out

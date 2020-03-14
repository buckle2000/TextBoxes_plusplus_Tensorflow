import tensorflow as tf
import pickle
from nets import np_methods
from pkg_resources import resource_filename
from collections import namedtuple

net_shape = (384, 384)

_model = tf.saved_model.load(resource_filename(__name__, 'saved_model'), tags=[tf.saved_model.SERVING])
_predict = _model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
_txt_anchors = pickle.load(open(resource_filename(__name__, 'anchors.pickle'), 'rb'))

def predict(img, select_threshold=0., nms_threshold=0.):
    """
    Detect scene text

    Parameters
    ----------
    img : np.ndarray with shape (None, None, 3) and dtype np.float32

    select_threshold : Only return results with score larger than this number

    nms_threshold : Threshold of non-maximum selection to bounding boxes

    Returns
    -------
    (classes, scores, bboxes) : tuple
        classes : shape=(N, ) class of objects predicted
        scores : shape=(N, ) confidence of prediction
        bboxes : shape=(N, 12) bounding box of objects
    """
    result = _predict(tf.convert_to_tensor(img))
    rbbox_img = result['bbox']
    rpredictions = [result[f"prediction_{i}"] for i in range(6)]
    rlocalisations = [result[f"localisation_{i}"] for i in range(6)]
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, _txt_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    # print(rscores)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(
        rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(
        rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
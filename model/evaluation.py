import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class CarousellEvaluation:

    def __init__(self, num_classes, class_mapping):
        if num_classes < 1:
            raise ValueError('Need at least 1 groundtruth class for evaluation.')
        self.num_classes = num_classes
        self._class_mapping = class_mapping
        self.gt_class_ids = np.empty(shape=(0,), dtype=np.int32)
        self.pred_class_ids = np.empty(shape=(0,), dtype=np.int32)

    def add_single_result(self, gt_class_ids, pred_class_ids):
        """
        Adds the ground truth info for the image

        :param np.ndarray gt_class_ids:   int32 numpy array of shape [num_boxes] containing the class ids for each TL
        :param np.ndarray gt_boxes:       int32 numpy array of shape [num_boxes, (y_min, x_min, y_max, x_max)]
                                          containing the bounding boxes for each TL
        """
        self.gt_class_ids = np.append(self.gt_class_ids, gt_class_ids, axis=0)
        self.pred_class_ids = np.append(self.pred_class_ids, pred_class_ids, axis=0)

    def evaluate(self):
        """
        Prints the classification report. And returns the F1-score, being the average per class of the F1-score.
        Returns:
            The F1-score
        """
        if self.gt_class_ids.shape[0] != self.pred_class_ids.shape[0]:
            raise ValueError(f'Both ground truth and predicted class ids should have the same length. Got '
                             f'{self.gt_class_ids.shape[0]} for ground truth and {self.pred_class_ids.shape[0]} '
                             f'for predicted')

        report = classification_report(self.gt_class_ids, self.pred_class_ids,
                                       labels=list(range(self.num_classes)),
                                       target_names=self._class_mapping.values())
        print(report)
        confusion_result = confusion_matrix(self.gt_class_ids, self.pred_class_ids, labels=list(range(self.num_classes)))
        print(confusion_result)
        return f1_score(self.gt_class_ids, self.pred_class_ids, average='micro')

    # recall = tf.keras.metrics.Recall(class_id=[0, 12])(labels, predictions)
    # recall_value = recall.result()
    # precision = tf.keras.metrics.Precision()
    # precision.update_state(labels, predictions)
    # precision_value = precision.result()
    # f1 = 2 * (recall_value * precision_value) / (recall_value + precision_value)
    # tf.print("recall", recall_value, "precision", precision_value)
    # tf.print("f1", f1)
    # for k in range(12):
    #     recall[k] = tf.keras.metrics.Recall()(
    #         labels=tf.equal(labels, k),
    #         predictions=tf.equal(predictions, k)).result()
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error


class CatEvaluation:
    def __init__(self, num_classes, class_mapping):
        if num_classes < 1:
            raise ValueError('Need at least 1 groundtruth class for evaluation.')
        self.num_classes = num_classes
        self._class_mapping = class_mapping
        self.gt_class_ids = np.empty(shape=(0,), dtype=np.int32)
        self.pred_class_ids = np.empty(shape=(0,), dtype=np.int32)

    def add_single_result(self, gt_class_ids, pred_class_ids):
        """
        Adds the ground truth and prediction info for the image

        :param np.ndarray gt_class_ids:   int32 numpy array of shape [num_class] containing the class ids for each image
        :param np.ndarray pred_class_ids:   int32 numpy array of shape [num_class] containing the class ids for each image
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


class PriceEvaluation:

    def __init__(self):
        self.gt_price = np.empty(shape=(0,), dtype=np.float32)
        self.pred_price = np.empty(shape=(0,), dtype=np.float32)

    def add_single_result(self, gt_price, pred_price):
        self.gt_price = np.append(self.gt_price, gt_price, axis=0)
        self.pred_price = np.append(self.pred_price, pred_price, axis=0)

    def evaluate(self):
        """
        Returns:
            The mean_absolute_error
        """
        return mean_absolute_error(self.gt_price, self.pred_price)

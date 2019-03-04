from PIL import Image
import numpy as np
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300

def main():
    pretrained_model = 'ssd300_voc0712_converted_2017_06_06.npz'
    model = SSD300(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model=pretrained_model)

    img = np.array(Image.open('cat.jpg'))
    img = img.transpose(2, 0, 1)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    result = []
    for b, lbl, s in zip(bbox, label, score):
        r = {'box': b.tolist(),
             'label': voc_bbox_label_names[lbl],
             'score': float(s)}
        result.append(r)
    print(result)

if __name__ == '__main__':
    main()
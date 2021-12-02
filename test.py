#from mmdet.apis import init_detector, inference_detector
#
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
## download the checkpoint from model zoo and put it in `checkpoints/`
## url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
#checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#device = 'cuda:0'
## init a detector
#model = init_detector(config_file, checkpoint_file, device=device)
## inference the demo image
#inference_detector(model, 'demo/demo.jpg')

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
#from mmdet.apis import show_result
import cv2
 
# 模型配置文件
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
 
# 预训练模型文件
checkpoint_file = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
 
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
# 测试单张图片并进行展示
img = cv2.imread('./demo/beauty.jpg')
#print(img.shape)
x, y = img.shape[0:2]
img_test1 = cv2.resize(img, (int(y / 1), int(x / 1))) #y/1以及x/1表示没有resize
result = inference_detector(model, img_test1 )
#show_result(img_test1, result, model.CLASSES)
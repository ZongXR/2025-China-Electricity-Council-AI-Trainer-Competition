# -*- coding: utf-8 -*-
import os
import paddle
import paddle.nn as nn
import numpy as np
from tqdm import tqdm
import typing
import cv2
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.core.workspace import create
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from ppdet.utils.logger import setup_logger
from ppdet.modeling.initializer import reset_initialized_parameter
import time

logger = setup_logger('ppdet.engine')


class Predictor:
    '''
    InitModel函数  模型初始化参数，注意不能自行增加删除函数入参
    Params:

    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    '''

    def __init__(self):
        self.trainer = None
        self.cfg = None

    def InitModel(self):
        ret = True
        err_message = "normal"
        '''
        模型初始化，由用户自行编写
        加载出错时，给ret和err_message赋值相应的错误

        例如
        ret=False
        err_message="model_path: [{}] init failed".format(model_path)
        '''
        # 输入配置文件和模型文件路径
        config_path = "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml"
        model_path = "output/ppyoloe_plus_crn_s_80e_coco/best_model.pdparams"
        # 读取配置文件
        if os.path.exists(config_path):
            self.cfg = load_config(config_path)
        else:
            return False, "配置文件读取失败，请检查文件是否正确"
        self.trainer = Trainer(self.cfg, mode='test')
        # 读取模型文件
        if os.path.exists(model_path):
            self.trainer.load_weights(model_path)
        elif os.path.exists(self.cfg.weights):
            self.trainer.load_weights(self.cfg.weights)
        else:
            return False, "模型文件读取失败，请检查文件是否正确"
        # 检查是否使用xpu进行预测服务
        # 默认不启用xpu进行预测，可在configs文件里进行配置
        if 'use_xpu' not in self.cfg:
            self.cfg.use_xpu = False
        if self.cfg.use_xpu:
            paddle.set_device('xpu')
        return ret, err_message

    '''
    Detect    模型推理函数，注意不能自行增加删除函数入参
    image_nparray cv2.imdecode(np.fromfile(pred_image_path,dtype=np.uint8),cv2.IMREAD_COLOR)读取后的图像
    return    列表[]
    '''

    def Detect(self, image_nparray):
        '''
        模型推理部分，由用户自行编写
        detect_result输出格式:
        [
            [category,xmin,ymin,xmax,ymax,score],
            [category,xmin,ymin,xmax,ymax,score],
            ....
            [category,xmin,ymin,xmax,ymax,score]
        ]
        数据格式:            示例
        category : str      dog      *类别名称必须与数据集中的类别名称一致
        xmin     : float    123.13
        ymin     : float    123.13
        xmax     : float    323.13
        ymax     : float    423.13
        score    : float    0.6666
        用户根据输出格式append到detect_result
        '''
        timestamp = str(time.time())
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

        # file_save_path = './tmp/test{}.jpg'.format(timestamp.replace('.','_'))
        single_image_file_path = './tmp/test{}.jpg'.format(timestamp.replace('.', '_'))
        cv2.imencode('.jpg', image_nparray, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(single_image_file_path)
        if not os.path.exists(single_image_file_path):
            print("[Error] img_file :[{}] not exists.".format(single_image_file_path))
            return []
        images = [single_image_file_path]
        detect_result = self.predict_nw(images)
        os.remove(single_image_file_path)
        return detect_result

    '''
    基于PaddleYOLO代码库系列的模型推理函数示例
    '''

    def predict_nw(self, images):
        self.trainer.dataset.set_images(images)
        loader = create('TestReader')(self.trainer.dataset, 0)

        imid2path = self.trainer.dataset.get_imid2path()

        # 读取coco标注数据，主要包含类别，及该类别对应的类别Id
        anno_file = self.trainer.dataset.get_anno()
        # clsid2catid, catid2name = get_categories(
        #     self.cfg.metric, anno_file=anno_file)
        clsid2catid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
        catid2name = {0: '080002', 1: '刀闸A打开母', 2: '刀闸A打开公', 3: '刀闸A触头母', 4: '刀闸A触头公', 5: '080003', 6: '刀闸B闭合触头',
                      7: '刀闸B打开横触头', 8: '刀闸B触头横触头', 9: '080001', 10: '刀闸A闭合触头', 11: '080004', 12: '刀闸B打开竖触头',
                      13: '刀闸B触头竖触头'}
        print('[INFO] catid2name = ', catid2name)

        # Run Infer
        self.trainer.status['mode'] = 'test'
        self.trainer.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.trainer.dataset, 0)
            self.trainer._flops(flops_loader)
        # 保存模型输出结果
        results = []
        test_batch_size = self.cfg.TestReader['batch_size']
        logger.info("Test loader length is {}, test batch_size is {}.".format(
            len(loader), test_batch_size))
        logger.info("Starting predicting ......\n")
        for step_id, data in enumerate(tqdm(loader)):
            self.trainer.status['step_id'] = step_id
            # forward
            outs = self.trainer.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # 将【模型输出结果】转换成【南网制定的标准输出结果】
        # img = cv2.imread("./dataset/test2.png")
        infer_results = []
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                self.trainer.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                    if 'bbox' in batch_res else None
                start = end

                # 借鉴visualize_results函数中的内容
                for dt in np.array(bbox_res):
                    if int(im_id) != dt['image_id']:
                        continue
                    catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
                    if score >= 0.011:
                        if len(bbox) == 4:
                            # draw bbox
                            # print(bbox)
                            xmin, ymin, w, h = bbox
                            xmin = max(xmin, 0)
                            ymin = max(ymin, 0)
                            w = max(w, 0)
                            h = max(h, 0)
                            xmax = xmin + w
                            ymax = ymin + h
                            # 加入到infer_results中
                            # cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),10)
                            infer_results.append([catid2name[catid], xmin, ymin, xmax, ymax, score])
                        elif len(bbox) == 8:
                            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                            xmin = min(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            xmax = max(x1, x2, x3, x4)
                            ymax = max(y1, y2, y3, y4)
                            # 加入到infer_results中
                            infer_results.append([catid2name[catid], xmin, ymin, xmax, ymax, score])
                        else:
                            logger.error('the shape of bbox must be [M, 4] or [M, 8]!')
        # cv2.imwrite('output/test1207.jpg',img)
        return infer_results


if __name__ == '__main__':
    ###备注说明：main函数提供给用户内测，修改后[不影响]评估
    predictor = Predictor()
    ret, err_message = predictor.InitModel()
    if ret:
        img_file = "/home/bml/storage/mnt/v-c6eb0ddf5bbd4/org/ds_model/24_01_sm/nw_data/train/img/DZA_F_1.jpg"
        image_nparray = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)  ## 支持中文，与cv2.imread读取一致
        detect_result = predictor.Detect(image_nparray)
        print("detect_result", detect_result)
        # for detect_result in detect_result:
        #     print(detect_result)
    else:
        print("[Error] InitModel failed. ret", ret, err_message)

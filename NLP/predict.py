# -*- coding: utf-8 -*-
# version:2024.8.8
# 2024.8月以后竞赛必须使用该版本
# 项目名称:客户标签画像分析
import os
import json

###请在这里import想调用的库
import argparse
import functools
import os

import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from utils import preprocess_function, read_local_dataset, read_texts_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger


class Predictor:
    '''
    InitModel函数  模型初始化参数,注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    '''

    def InitModel(self):
        ret = True
        err_message = "normal"
        '''
        模型初始化,由用户自行编写
        加载出错时给ret和err_message赋值相应的错误
        *注意模型应为相对路径
        '''
        ### 请在try内编写模型初始化,便于捕获错误
        try:
            ##########模型初始化开始##########
            ### 加载模型
            ### self.model=load_model(model_path)
            self.device = "xpu"
            self.model = AutoModelForSequenceClassification.from_pretrained("./checkpoint")
            self.tokenizer = AutoTokenizer.from_pretrained("./checkpoint")
            self.max_seq_length = 512
            self.batch_size = 1

            self.label_list = []
            label_path = os.path.join("./dataset", "label.txt")
            with open(label_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    self.label_list.append(line.strip())
            ##########模型初始化结束##########
        except Exception as err:  ### 此行不能删除
            ret = False  ### 此行不能删除
            err_message = "[Error] model init failed,err_message:[{}]".format(ExceptionMessage(err))  ### 此行不能删除
            print(err_message)

        return ret, err_message

    '''
    Detect         模型推理函数,注意不能自行增加删除函数入参
    text           输入单个文本,非批量
    return         字典dict
    '''

    def Detect(self, text):
        '''
            模型推理部分,由用户自行编写
            detect_result输出格式示例:
            {
                "predict1_code":"a01",
                "predict2_code":"b01",
                "predict3_code":"c01",
                "predict4_code":"",
                "predict5_code":""
            }
            数据格式说明:
            1.字典中key:predict1-5_code分别表示 一级标签到五级标签的标签编码
            2.字典中value均为由标签编码,类型为字符串,例如"a01",标签编码请参考比赛文档
            3.每一级标签仅有1种标签,默认缺省时为""
        '''
        ### 请在try内编写推理代码,便于捕获错误
        try:
            ##########模型推理开始##########
            detect_result = {
                "predict1_code": "",
                "predict2_code": "",
                "predict3_code": "",
                "predict4_code": "",
                "predict5_code": ""
            }
            if text is None:
                print("[Error] text is None.")
            # data = {"text": text}
            # print(text)
            # for batch in data:
            result = self.predict(text)
            for i in result.keys():
                if i == 0:
                    detect_result["predict1_code"] = result[i][0]
                elif i == 1:
                    detect_result["predict2_code"] = result[i][0]
                elif i == 2:
                    detect_result["predict3_code"] = result[i][0]
                elif i == 3:
                    detect_result["predict4_code"] = result[i][0]
                elif i == 4:
                    detect_result["predict5_code"] = result[i][0]
                    # lalel_i =
                # print(result[i][0])
                # print(type(result[i]))
            # print(result)
            return detect_result

            # return detect_result
            ##########模型推理结束##########
        except Exception as err:  ### 此行不能删除
            print("[Error] predictor.Detect failed.err_message:{}".format(ExceptionMessage(err)))  ### 此行不能删除
            return err  ### 此行不能删除

    # @paddle.no_grad()
    def predict(self, data):
        """
        Predicts the data labels.
        """

        data_ds = load_dataset(
            read_texts_dataset, texts=[data], is_test=True, lazy=False
        )
        # with open(os.path.join(args.dataset_dir, args.data_file)) as f:
        #     texts = f.readlines()
        #
        # data_ds = load_dataset(
        #     read_texts_dataset, texts=texts, is_test=True, lazy=False
        # )

        trans_func = functools.partial(
            preprocess_function,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            label_nums=len(self.label_list),
            is_test=True,
        )

        data_ds = data_ds.map(trans_func)
        # batchify dataset
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        data_batch_sampler = BatchSampler(data_ds, batch_size=self.batch_size, shuffle=False)

        data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)
        results = []
        self.model.eval()
        for batch in data_data_loader:
            logits = self.model(**batch)
            probs = F.sigmoid(logits).numpy()
            # logger.info(probs)
            for prob in probs:
                labels = []
                for i, p in enumerate(prob):
                    # logger.info(p)
                    if p > 0.1:
                        labels.append(self.label_list[i])

                results.append(labels)
            for t, labels in zip(data_ds.data, results):
                hierarchical_labels = {}
                # logger.info("text: {}".format(t["sentence"]))
                # logger.info("prediction result: {}".format(",".join(labels)))
                for label in labels:
                    for i, l in enumerate(label.split("##")):
                        if i not in hierarchical_labels:
                            hierarchical_labels[i] = []
                        if l not in hierarchical_labels[i]:
                            hierarchical_labels[i].append(l)
                # for d in range(len(hierarchical_labels)):
                # logger.info("level {} : {}".format(d + 1, ",".join(hierarchical_labels[d])))
                # logger.info("--------------------")
        return hierarchical_labels


### 获取异常文件+行号+信息
def ExceptionMessage(err):
    err_message = (
            str(err.__traceback__.tb_frame.f_globals["__file__"])
            + ":"
            + str(err.__traceback__.tb_lineno)
            + "行:"
            + str(err)
    )
    return err_message


if __name__ == '__main__':
    ###备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor = Predictor()
    ret, err_message = predictor.InitModel()
    if ret:
        text = r"""微信用户 $ 你好，我想咨询电费相关的问题。\n微信用户 $ 转人工\n高东丽 $ 新年好,请问有什么可以帮您？\n微信用户 $ 如何添加用电人数\n高东丽 $ 您好，请问是需要办理一户多人****？\n微信用户 $ 是的\n高东丽 $ 一户多人口业务微信服务号****路径：关注“南网在线”微信服务号并绑定您的用户编号＞点击“我的用电”＞“业务办理”＞“业务变更”＞“一户多人口”进入或从“南网在线”微信小程序首页，点击【一…\n高东丽 $ 您好，您可以按照以上指引进行申请	咨询查询##业务流程、进度##改类
^o^海^0^阔天空 $ 电费交了双重了\n王泽天 $ 您好，您已接入在线客服，很高兴为您服务\n^o^海^0^阔天空 $ 怎样申请退款\n王泽天 $ 请问具体是哪一户电表，为了保障客户信息安全。请您提供用户编号，这边为您核实查询\n^o^海^0^阔天空 $ 有一个户交双重电费了\n^o^海^0^阔天空 $ img null http\n^o^海^0^阔天空 $ 您尾号8561卡8月1****:07工商银行支出(电费广州)227.47元，余额18,998.77元。【工商银行】\n^o^海^0^阔天空 $ 0800840****3****\n^o^海^0^阔天空 $ 这个户交了双重电费了\n^o^海^0^阔天空 $ ???\n王泽天 $ 非常抱歉，由于微信于2023年8月9-24日期间进行代扣系统升****作，导致您户出现重复扣费情况，目前微信方正紧急修复当中，经查询已自动为您转为预收****月电费，请您耐心等待，给您造成…\n^o^海^0^阔天空 $ ??\n王泽天 $ 您好，经查询目前重复收费的电费****为预收款抵扣下月电费，给您造成不便，深感抱歉"""
        detect_result = predictor.Detect(text)
        print("detect_result", detect_result)
    else:
        print("[Error] InitModel failed. ret", ret, err_message)

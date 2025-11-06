# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/bml/storage/aifs/public/data_gaixc")
import DataGaixc
#数据集ID
datasetId="5dedbd63759a4768937d4af13b33e182"
#数据集版本
dversion="V1"
#下载数据保存目录
fileSavePath="download"
DataGaixc.DownloadDataset(datasetId,dversion,fileSavePath)

#数据集ID
datasetId="5dedbd63759a4768937d4af13b33e182"
#数据集版本
dversion="V2"
#下载数据保存目录
fileSavePath="download"
DataGaixc.DownloadDataset(datasetId,dversion,fileSavePath)

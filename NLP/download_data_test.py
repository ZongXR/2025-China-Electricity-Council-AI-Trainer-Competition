# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/bml/storage/aifs/public/data_gaixc")
import DataGaixc
#数据集ID
datasetId="27baea4bbd894c8783d2f924ac89a1aa"
#数据集版本
dversion="V2"
#下载数据保存目录
fileSavePath="download"
DataGaixc.DownloadDataset(datasetId,dversion,fileSavePath)

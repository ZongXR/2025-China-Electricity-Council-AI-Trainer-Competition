import sys
sys.path.append("/home/bml/storage/aifs/public/data_gaixc")
import DataGaixc
#需要上传的模型文件的本地地址
file_path="save_result.tar"
DataGaixc.UploadModel(file_path)

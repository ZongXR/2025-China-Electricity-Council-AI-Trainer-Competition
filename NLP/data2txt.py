import pandas as pd
import random
import os
import re
import argparse

label_columns = ["一级标签编码", "二级标签编码", "三级标签编码", "四级标签编码", "五级标签编码"]


def combine_labels(row):
    labels = []
    for col in label_columns:
        if pd.notna(row[col]):
            labels.append(str(row[col]))
    return "##".join(labels) if labels else ''


def daset_build(dataset_dir="./", output_dir="./data", excel_input_file="", train_proportion=0.8, val_proportion=0.2):
    excel_df_all = pd.DataFrame()
    if excel_input_file != "":
        file_path = os.path.join(dataset_dir, excel_input_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        excel_df_all = pd.read_excel(file_path, sheet_name="Sheet1")
        rows, cols = excel_df_all.shape
        print(f"行数:{rows},列数:{cols}")
        excel_df_all['标签组合'] = excel_df_all.apply(combine_labels, axis=1)
    else:
        for file_name in os.listdir(dataset_dir):
            excel_path = os.path.join(dataset_dir, file_name)
            # print(os.path.splitext(os.path.basename(excel_path))[1])
            if os.path.splitext(os.path.basename(excel_path))[1] == ".xlsx":
                print(excel_path, "正在解析...")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                excel_df = pd.read_excel(excel_path, sheet_name="Sheet1")
                rows, cols = excel_df.shape
                print(f"行数:{rows},列数:{cols}")
                excel_df['标签组合'] = excel_df.apply(combine_labels, axis=1)
                excel_df_all = pd.concat([excel_df_all, excel_df], ignore_index=True, axis=0)
    # 去除消息详情的空白字符
    excel_df_all["消息详情"] = excel_df_all["消息详情"].str.replace(r'\s+', '', regex=True)
    excel_df_all = excel_df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    data_str = excel_df_all["消息详情"] + "\t" + excel_df_all["标签组合"] + "\n"

    total_data = len(data_str)
    train_data = int(len(data_str) * train_proportion)
    dev_data = int(len(data_str) * val_proportion)
    train_txt = os.path.join(output_dir, "train.txt")
    with open(train_txt, "w") as f:
        for data in data_str[:train_data]:
            f.write(str(data))
        print("train.txt has been generated....")
    # 制作dev.txt
    val_txt = os.path.join(output_dir, "dev.txt")
    with open(val_txt, "w") as f:
        for data in data_str[train_data:train_data + dev_data]:
            f.write(str(data))
        print("val.txt has been generated....")
    # 制作test.txt
    test_txt = os.path.join(output_dir, "test.txt")
    with open(test_txt, "w") as f:
        for data in data_str[train_data + dev_data:]:
            f.write(str(data))
        print("val.txt has been generated....")
    # 制作lebel.txt
    label_txt = os.path.join(output_dir, "label.txt")
    with open(label_txt, "w") as f:
        for label in excel_df_all['标签组合'].drop_duplicates():
            f.write(str(label) + "\n")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_dir',
        help='datasets directory',
        type=str,
        default='./')
    parser.add_argument(
        '--output_dir',
        help='datasets directory',
        type=str,
        default='./data')
    parser.add_argument(
        '--excel_input_file',
        help='input annotated directory',
        type=str,
        default=''
    )
    parser.add_argument(
        '--train_proportion',
        help='the proportion of train dataset',
        type=float,
        default=0.8)
    parser.add_argument(
        '--val_proportion',
        help='the proportion of train dataset',
        type=float,
        default=0.2)
    args = parser.parse_args()

    daset_build(args.dataset_dir, args.output_dir, args.excel_input_file, args.train_proportion, args.val_proportion)


if __name__ == "__main__":
    main()

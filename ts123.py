import csv
import os
def save_parameters_to_csv(filename, parameters):
    # parameters 是一个字典，键是行名称，值是参数值
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ["porprotion", "count", "d", "coherence", "perplexity"]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        if os.stat(filename).st_size == 0:
            csv_writer.writeheader()

        # 写入参数值
        csv_writer.writerow(parameters)

# 示例参数
parameters = {
    "porprotion": 0.5,
    "count": 100,
    "d": 3,
    "coherence": 0.8,
    "perplexity": 20
}
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# 在文件名中添加时间戳
filename_with_timestamp = f"parameters_{current_time}.csv"
# 保存到CSV文件
save_parameters_to_csv("parameters.csv", parameters)

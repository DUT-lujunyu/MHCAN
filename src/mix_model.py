import json

root_path = "../results/"
# root_path = "../"

all_res = []
file1 = [2,8,4,14,13,12]
for i in file1:
    path = root_path + str(i) + ".json"
    with open(path, 'r') as file:
        res = json.load(file)
    all_res.append(res)

# print(all_res[0][0])

super_res = []
for i in range(len(all_res[0])):  # 第i条样本
    dic = {}
    dic["title"] = all_res[0][i]["title"]
    dic["abstract"] = all_res[0][i]["abstract"]

    all_preds = {}
    for index in range(len(file1)):  # 一条数据的n个预测结果, index是文件序号
        preds = all_res[index][i]["pred_labels"]
        for pred in preds:
            if pred in all_preds:
                all_preds[pred] += 1
            else:
                all_preds[pred] = 1
    
    final_pred = []
    for key, value in all_preds.items():
        if value >= 3:
            final_pred.append(key)
    dic["pred_labels"] = final_pred
    super_res.append(dic)

print(super_res[0])
with open("../test.json", 'w') as f:
    json.dump(super_res, f)

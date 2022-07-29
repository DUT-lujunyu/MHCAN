import json
import re

file = "test_v2.json"
# file = "verification set.json"

title_list = []
abstract_list = []
level1_list = []
level2_list = []
level3_list = []
levels_list = []
data_new = []

with open(file, 'r') as f:
    data = json.load(f)

print("Load file successfully.")

for d in data:
    title_list.append(d.get("title"))
    abstract_list.append(d.get("abstract"))
    level1_list.append(d.get("level1"))
    level2_list.append(d.get("level2"))
    level3_list.append(d.get("level3"))
    levels_list.append(d.get("levels"))

for i in range(0, len(data)):
    title_list[i] = re.sub(
        "[αβγδεζηθικλμνξοπρστυφχψω†=→@*‘+]+", "", title_list[i])  # 希腊字母与特殊符号
    title_list[i] = re.sub("\\<[^>]*\\>", "", title_list[i])  # 尖括号及其内容
    title_list[i] = re.sub("\\([^)]*\\)", "", title_list[i])  # 圆括号及其内容
    title_list[i] = re.sub("\\[[^]]*\\]", "", title_list[i])  # 方括号及其内容
    title_list[i] = re.sub("\\\\u\d{4}", "", title_list[i])  # 特殊unicode

    abstract_list[i] = re.sub(
        "[αβγδεζηθικλμνξοπρστυφχψω†=→@*‘+]+", "", abstract_list[i])
    abstract_list[i] = re.sub("\\<[^>]*\\>", "", abstract_list[i])
    abstract_list[i] = re.sub("\\([^)]*\\)", "", abstract_list[i])
    abstract_list[i] = re.sub("\\[[^]]*\\]", "", abstract_list[i])
    abstract_list[i] = re.sub("\\\\u\d{4}", "", abstract_list[i])

    data_new.append(dict(title=title_list[i], abstract=abstract_list[i], level1=level1_list[i],
                    level2=level2_list[i], level3=level3_list[i], levels=levels_list[i]))


with open(file[:-5] + "_modified.json", 'w') as f:
    json.dump(data_new, f)

print("Finished.")

'''
化学式（有bug，会把大写字母开头的单词也匹配到，暂时去除）
[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+

'''

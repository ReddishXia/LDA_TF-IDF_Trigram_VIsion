import re

word = 'high-sensitivitycardiacTroponinT'
pattern_str = 'hs-cTnT'

# 将目标单词转换为列表
word_list = list(word)

# 将待匹配的字符串转换为正则表达式
pattern = re.compile(''.join(['[{}]'.format(c) for c in pattern_str]))

# 在目标单词列表中匹配正则表达式中的每个字母
matches = [c for c in word_list if pattern.match(c)]

# 判断是否全部匹配成功
if len(matches) == len(pattern_str):
    print("目标单词中包含该字符串所有的字母，顺序和出现次数一一对应")
else:
    print("目标单词中不包含该字符串所有的字母")

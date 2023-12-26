list1 = ['Multiple Sclerosis (MS)', 'nuclear factor erythroid 2 related factor 2 (Nrf2)', 'psoriasis', 'hospitalisation', 'calming', 'dimethyl fumarate DMF ', 'HCAR2 activation ', 'NADPH quinone oxidoreductase 1 (NQO1)', ' (SOD1)', 'cytoprotective']
list2 = ['immune dysregulated hyperinflammation ', 'NRF2', 'oxidative stresses ', 'hospitalisation', 'calming', 'multiple sclerosis ', 'anti-oxidant drugs ', 'dimethyl fumarate DMF ', 'HCAR2 activation ', 'NQO1']

# 转换为集合
set1 = set(list1)
set2 = set(list2)

# 找出不同的单词
different_words = set1.symmetric_difference(set2)

# 输出结果
print("第一个列表中的不同单词：", different_words.intersection(set1))
print("第二个列表中的不同单词：", different_words.intersection(set2))
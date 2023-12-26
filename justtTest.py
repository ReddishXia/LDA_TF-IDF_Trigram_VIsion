import os
import numpy as np




rootPath = r"C:\Users\79988\PycharmProjects\chatGPT_LDA_Trigram\test"
xml_papers = os.listdir(rootPath)
index_dict=[['PMC7830522.xml', 'PMC9281590.xml', 'PMC9284567.xml', 'PMC9287470.xml', 'PMC9289664.xml', 'PMC9298711.xml', 'PMC9315516.xml', 'PMC9329729.xml'], ['PMC7824470.xml', 'PMC7825841.xml', 'PMC7825955.xml', 'PMC7826042.xml', 'PMC7828218.xml', 'PMC7828525.xml', 'PMC7830475.xml', 'PMC7915126.xml', 'PMC8238037.xml', 'PMC9278083.xml', 'PMC9295679.xml', 'PMC9302868.xml', 'PMC9314237.xml', 'PMC9329705.xml', 'PMC9330984.xml', 'PMC9334668.xml', 'PMC9337782.xml', 'PMC9338729.xml', 'PMC9380879.xml'], ['PMC7824170.xml', 'PMC7829938.xml', 'PMC9326243.xml', 'PMC9327186.xml', 'PMC9335408.xml'], ['PMC8504968.xml', 'PMC9298468.xml', 'PMC9308500.xml', 'PMC9328125.xml', 'PMC9334659.xml', 'PMC9338774.xml'], ['PMC7323196.xml', 'PMC7829836.xml', 'PMC8524328.xml', 'PMC9329661.xml', 'PMC9338857.xml'], ['PMC7824817.xml', 'PMC7831030.xml', 'PMC7831046.xml', 'PMC7831568.xml', 'PMC9277988.xml', 'PMC9290425.xml', 'PMC9291241.xml', 'PMC9298551.xml', 'PMC9309232.xml', 'PMC9310363.xml', 'PMC9312374.xml', 'PMC9312909.xml', 'PMC9313162.xml', 'PMC9326076.xml', 'PMC9329529.xml', 'PMC9329707.xml', 'PMC9334857.xml', 'PMC9336601.xml', 'PMC9337784.xml', 'PMC9338349.xml'], ['PMC7824075.xml', 'PMC7824811.xml', 'PMC7827974.xml', 'PMC7830623.xml', 'PMC9326261.xml', 'PMC9335482.xml'], ['PMC7405836.xml', 'PMC7827890.xml', 'PMC7830673.xml', 'PMC9329687.xml'], ['PMC7827846.xml', 'PMC7829816.xml', 'PMC7830154.xml', 'PMC8499788.xml', 'PMC9242884.xml', 'PMC9295384.xml', 'PMC9312732.xml', 'PMC9322618.xml', 'PMC9334848.xml', 'PMC9335884.xml', 'PMC9338730.xml'], ['PMC9289581.xml', 'PMC9327440.xml', 'PMC9337791.xml'], ['PMC9242688.xml', 'PMC9281458.xml', 'PMC9283122.xml', 'PMC9296900.xml', 'PMC9298167.xml', 'PMC9335480.xml'], [], [], ['PMC8755319.xml', 'PMC9043892.xml', 'PMC9278178.xml', 'PMC9307220.xml'], ['PMC7827130.xml', 'PMC7828126.xml', 'PMC9284599.xml', 'PMC9295334.xml', 'PMC9308998.xml', 'PMC9311342.xml', 'PMC9326230.xml', 'PMC9326247.xml', 'PMC9328743.xml', 'PMC9333103.xml'], ['PMC7827692.xml', 'PMC7828742.xml', 'PMC9285112.xml', 'PMC9328841.xml', 'PMC9329734.xml'], ['PMC7828055.xml', 'PMC7830668.xml', 'PMC7831024.xml', 'PMC7831665.xml'], ['PMC7825705.xml', 'PMC7831445.xml'], ['PMC7829843.xml', 'PMC9325633.xml']]
xd=['PMC7824075.xml', 'PMC7824170.xml', 'PMC7824470.xml', 'PMC7824811.xml', 'PMC7824817.xml', 'PMC7825705.xml', 'PMC7825841.xml', 'PMC7825955.xml', 'PMC7826042.xml', 'PMC7827130.xml', 'PMC7827692.xml', 'PMC7827846.xml', 'PMC7827890.xml', 'PMC7827974.xml', 'PMC7828055.xml', 'PMC7828126.xml', 'PMC7828218.xml', 'PMC7828525.xml', 'PMC7828742.xml', 'PMC7829816.xml', 'PMC7829836.xml', 'PMC7829843.xml', 'PMC7829938.xml', 'PMC7830154.xml', 'PMC7830475.xml', 'PMC7830522.xml', 'PMC7830623.xml', 'PMC7830668.xml', 'PMC7830673.xml', 'PMC7831024.xml', 'PMC7831030.xml', 'PMC7831046.xml', 'PMC7831445.xml', 'PMC7831568.xml', 'PMC7831665.xml']
positions = [xml_papers.index(item) for item in xd]

print(positions)
result = [np.where(row < 0.1)[0].tolist() for row in doc_topic_dists]

for i, item in enumerate(xd):
    mywords_no_stopwrd = []
    mywords_no_stopwrd.append(item)
    for j ,item1 in enumerate(result[positions[i]]):
        mywords_no_stopwrd+=index_dict[item1]
    print(mywords_no_stopwrd)




for i, item in enumerate(index_dict):
    if not item:
        print("empty_dict is empty")
    else:
        mywords_no_stopwrd = []
        for j, aricleNum in enumerate(item):
            if aricleNum in xd:
                mywords_no_stopwrd.append(aricleNum)
        for k, item3 in enumerate(index_dict):
            if k!=i:
                mywords_no_stopwrd+=item3

    print(mywords_no_stopwrd)

import numpy as np

doc_topic_dists = np.array([[0.2, 0.3, 0.01],
                           [0.1, 0.4, 0.5],
                           [0.6, 0.02, 0.7]])

# 使用布尔索引和列表推导式找到小于0.1的位置
result = [np.where(row < 0.1)[0].tolist() for row in doc_topic_dists]

print(result)

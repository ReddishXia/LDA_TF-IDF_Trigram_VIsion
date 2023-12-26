from itertools import chain
import justTest as js
def getFreq_Term(term,freq,Total,categories):
    class MyClass:
        def __init__(self, category, total, term, freq):
            self.category = category
            self.total = total
            self.term = term
            self.freq = freq


    # grouped_terms = {}
    # for topic, term in zip(categories , term):
    #     if topic not in grouped_terms:
    #         grouped_terms[topic] = []
    #     grouped_terms[topic].append(term)

    category_dict = {}
    for cat, tot, term, freq in zip(categories, Total, term, freq):
        if cat not in category_dict:
            category_dict[cat] = []

        category_dict[cat].append(MyClass(cat, tot, term, freq))
    for cat, items in category_dict.items():
        sorted_items = sorted(items, key=lambda x: x.freq, reverse=True)
        category_dict[cat] = sorted_items
    count = 0
    # 打印结果
    for cat, items in category_dict.items():
        print(f'Category: {cat}')
        max_Total=0.0
        for item in items:
            if item.total>max_Total:
                max_Total=item.total

        for item in items[:30]:
            if(js.getProportion(item.freq,item.total,max_Total)>0.5):
                count+=1
        # print(count)
            # print(f'Total: {item.total}, Term: {item.term}, Freq: {item.freq}')
        # print('-' * 20)
    return count

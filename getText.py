import os
import platform
import bioc
import time
from numba import jit
t1 = time.time()
currentPath = os.path.dirname(os.path.realpath(__file__))
host_os = platform.system()
if host_os == 'Windows':
    rootPath = currentPath+'\\'+'test'
my_xml_papers = os.listdir(rootPath)
print(rootPath+ '\\')
dic = {}
def To_Generate_All(input):
    number = input
    # print(rootPath + '\\' + f'PMC{number}.xml')
    # try:
    reader = bioc.biocxml.BioCXMLDocumentReader(rootPath + '\\' + f'PMC{number}.xml')
    # except SyntaxError:
    #     pass

    article = [x for x in reader][0]
    # print(article.id)
    # print(article.passages)
    # get "section_type" and "type"
    # --for an individual passage. For example, #10
    # ----get "section_type" individually
    dct = article.passages[10].infons
    value_iterator = iter(dct.values())
    section_type = next(value_iterator)  # the first value
    # print(section_type)
    # ----get both "section_type" and "type"
    first, second = list(dct.values())[:2]
    # print(first, second)
    # --list all
    # ----list all "section_type"
    section_types = [next(iter(x.infons.values())) for x in article.passages]
    # print(section_types)
    # ----list all "section_type" and "type"
    allTypes = [list(x.infons.values())[:2] for x in article.passages]
    # print(allTypes)

    # get text
    # --for an individual passage. For example, #10
    # print(article.passages[10].text)
    # --list all text with certain "section_type". For example, "INTRO"
    allText = []
    text1 = ""
    for i in article.passages:
        if next(iter(i.infons.values())) in ['ABSTRACT', 'CONCL', 'METHODS', 'RESULTS']:
            judgeShort = 1

        if next(iter(i.infons.values())) in ['ABSTRACT', 'CONCL']:
            text1+=i.text.replace('\u2009', ' ')
            allText.append(i.text.replace('\u2009', ' ')+" "+i.text.replace('\u2009', ' '))

        else:
            allText.append(i.text.replace('\u2009', ' '))
    # print(allText)
    # --list all text with certain "type". For example, "paragraph"
    allText2 = []
    text = ""
    for i in article.passages:
        if list(i.infons.values())[:2][1] == "paragraph":
            text += i.text.replace('\u2009', ' ')
            allText2.append(i.text.replace('\u2009', ' '))
    text=text1+text
    return text


def getArtcle():
    for xml_name in my_xml_papers:
        # print(xml_name[3:-4])
        try:
            dic[xml_name]=To_Generate_All(xml_name[3:-4])
        except SyntaxError:
            pass
    return dic
t2 = time.time()
print(int(t2-t1))
i=0
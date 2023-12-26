import re

abbreviation = "CH-HM"
text = "capital (Hospitales de Madrid del Grupo HM Hospitales (CH-HM)"
def return_MinMainString(text,str,pattern_First_Positions):
    for i in range(len(text)):
        if text[i] == str[0]:
            pattern_First_Positions.append(i)
    pattern_First_Positions=pattern_First_Positions[::-1]

    for num,position in  enumerate(pattern_First_Positions):
        match_Len=0
        i=0
        for x in range(position, len(text)):
            if text[x]==str[i]:
                match_Len+=1
                i += 1
        if match_Len==len(str):
            return num
    return -1
def return_MainString(text,str):
    lower_phrase = text.lower()
    match_Len=0
    i=0
    for x in range(0, len(lower_phrase)):
        if lower_phrase[x]==str[i]:
            match_Len+=1
            i += 1
            if match_Len==len(str):
                return 1
    return -1

def get_remaining_sentences(phrase, k,abv):

    phrase = re.sub('/', ' ', phrase)
    phrase = re.sub('-', ' ', phrase)
    abv = re.sub('-', '', abv)
    lower_phrase = phrase.lower()
    words = phrase.split(' ')
    lower_words=lower_phrase.split(' ')
    count = 0
    start_index = -1

    for i, word in enumerate(lower_words):
        if word.startswith(abv[0]):
            count += 1
            if count == k:
                start_index = i + 1
                break

    if start_index != -1:
        remaining_sentence = ' '.join(words[start_index-1:])
        return remaining_sentence
    else:
        return ""

class Solution:
    # 获取next数组
    def get_next(self, T):
        i = 0
        j = -1
        next_val = [-1] * len(T)
        while i < len(T) - 1:
            if j == -1 or T[i] == T[j]:
                i += 1
                j += 1
                # next_val[i] = j
                if i < len(T) and T[i] != T[j]:
                    next_val[i] = j
                else:
                    next_val[i] = next_val[j]
            else:
                j = next_val[j]
        return next_val
    # KMP算法
    def kmp(self, S, T):
        i = 0
        j = 0
        next = self.get_next(T)
        while i < len(S) and j < len(T):
            if j == -1 or S[i] == T[j]:
                i += 1
                j += 1
            else:
                j = next[j]
        if j == len(T):
            return i - j
        else:
            return -1
def merge_sublist_to_word(words, k, length):
    sublist = words[k:k + length]  # 获取从索引k开始，长度为length的子列表
    merged_word = ' '.join(sublist)  # 将子列表中的单词合并成一个单词

    return merged_word

def find_abbreviation(text, abbreviation):

    text_lower = text.lower()
    abbreviation_lower = abbreviation.lower()
    # 查找缩写在文本中的位置
    index = text_lower.find("(" + abbreviation_lower)
    real_abbreviation = ""
    if index != -1:
        while (text[index] != ")"):
            real_abbreviation += (text[index])
            index += 1

    if index != -1:
        return real_abbreviation
    else:
        return 0

def find_full_name(text, abbreviation):

    if any(char.isupper() for char in abbreviation):
        # 通过正则表达式匹配MINJ所在的短语
        length=len(abbreviation)
        match = re.search(r'\b(\w+\W+){0,%d}\(+%s\b' % (length*2, abbreviation), text)
        if match:
            print("yes")
        else:
            if find_abbreviation(text,abbreviation)!=0:
                abbreviation=find_abbreviation(text,abbreviation).replace("(","")
                match = re.search(r'\b(\w+\W+){0,%d}\(+%s\b' % (length * 2, abbreviation), text)
                length = len(abbreviation)
            else:
                return abbreviation

        # 获取匹配到的短语，并将其中的括号和MINJ去掉
        if match:
            phrase = match.group(0).replace('('+abbreviation, '').strip()

            clean_phrase = re.sub('-', ' ', phrase)
            clean_phrase = re.sub('/', ' ', clean_phrase)
            clean_phrase = re.sub(r'[^\w\s]', '', clean_phrase)
            # 获取短语中的前四个单词
            words = clean_phrase.split()[:length*2]
            initials = ''.join(word[0] for word in words)
            pattern=abbreviation.lower()
            s = Solution()
            start_situation=s.kmp(initials.lower(), pattern)
            if start_situation==-1:
                part2_words = clean_phrase.split(" ")
                initials = ''.join(word[0].lower() for word in part2_words)
                pattern_First_Positions = []
                num = return_MinMainString(initials, pattern, pattern_First_Positions)
                if num != -1:
                    orignal_Position = len(pattern_First_Positions) - num
                    return get_remaining_sentences(phrase, orignal_Position,pattern).replace("(","")+" ("+abbreviation+")"
                #第三种情况
                else:
                    pattern_First_Positions=pattern_First_Positions[::-1]
                    for num,position in enumerate(pattern_First_Positions):
                        orignal_Position = len(pattern_First_Positions)-num
                        less_FullName=get_remaining_sentences(phrase, orignal_Position,pattern)
                        less_FullNameString = ''.join(less_FullName).replace(" ","")
                        if return_MainString(less_FullNameString,pattern.replace("-",""))==1:
                            return less_FullName.replace("(","")+" ("+abbreviation+")"
                        n=0
            merged_word = merge_sublist_to_word(words, start_situation, length)
            return merged_word.replace("(","")+" ("+abbreviation+")"
        return abbreviation
    return abbreviation

print(find_full_name(text, abbreviation))

#
#
#
#         # words.reverse()  # 翻转列表
#         try:
#             index =  words.index(next(word for word in words if word.startswith(abbreviation[0]))) +1
#         except StopIteration:
#             print("")
#         # 找到第一个首字母为'M'的单词在列表中的位置
#         result = words[:index]  # 输出在这个单词后的所有单词
#         result.reverse()  # 翻转列表
#         s = ''.join(result)
#         s_set = [char for char in s.lower()]
#
#         target_set = [char for char in abbreviation.lower()]
#
#         index=0
#         intersect = []
#         for letter in target_set:
#             for letter1 in range(index, len(s_set)):
#                 index+=1
#                 if letter in s_set:
#                     intersect.append(letter)
#                     break
#         # 判断交集的长度是否大于等于 4
#         if len(intersect) == len(abbreviation):
#             fullName = ' '.join(result)
#             print(fullName)
#             return fullName
#
#  # ['Myocardial', 'injury']
#
# print(find_full_name(text, abbreviation))
#
# #CHAT

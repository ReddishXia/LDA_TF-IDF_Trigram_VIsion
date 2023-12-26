import numpy as np

def map_freq_to_width(freq_value, max_total, barwidth):
    return (freq_value / max_total) * barwidth

# 示例数据

# 获取 'Total' 属性的最大值
def getProportion(freq,Total,maxToyal):
    # 假设有一个数据对象
     # 这里的 305.31540010461646 是示例数据，请根据实际情况替换

    # 将数据对象中的 Freq 属性值映射到宽度值
    redBar_width_value = map_freq_to_width(freq, maxToyal, 530)
    grayBar_width_value= map_freq_to_width(Total, maxToyal, 530)
    return(redBar_width_value/grayBar_width_value)


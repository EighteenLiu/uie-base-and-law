# 单个案例分析
from pprint import pprint
from paddlenlp import Taskflow

# 定义schema，分开原告和被告的身份信息
schema = ['原告个人', '原告企业']

# 创建信息抽取任务，加载指定的checkpoint
ie = Taskflow('information_extraction',
              checkpoint='D:/python/bert/数据法治/checkpoint1/model_best/checkpoint-320',
              schema=schema)

# 修改代码实现从文件中逐行获取文本并分析
# 打开文件
with open('D:/桌面/text.txt', 'r', encoding='utf-8') as f:
    # 逐行读取
    for line in f:
        # 执行信息抽取
        results = ie(line)
        # 格式化输出结果
        #pprint(results)
        # 只将results中'原告-企业':后面[]里面的内容写入乐out.txt文件中
        index = 0
        if isinstance(results, list) and len(results) > index:
            with open('D:/桌面/textout.txt', 'a', encoding='utf-8') as f:
                f.write(str(results[index]) + '\n')
        else:
            print("Error: results 不是一个列表或索引超出范围")

# 关闭文件
f.close()

# 从乐out.txt文件中逐行读取文本，将原告-个人或者原告-企业中的原告姓名提取出来写入新建字典中，并统计原告姓名出现的次数
# 打开文件，并清空文件内容
with open('D:/桌面/textout.txt', 'r', encoding='utf-8') as f:
    # 新建字典
    name_dict = {}
    # 逐行读取
    for line in f:
        # 提取原告-个人或者原告-企业中的原告姓名，只提取'text':后面''里面的原告姓名
        if '原告个人' in line or '原告企业' in line:
            name = line.split("'text': '")[1].split("'")[0]

            # 统计原告姓名出现的次数
            if name in name_dict:
                name_dict[name] += 1
            else:
                name_dict[name] = 1
    # 格式化输出结果
    pprint(name_dict)
    #从name_dict字典中找出出现次数大于10的原告姓名并将姓名输出到屏幕上
    for key, value in name_dict.items():
        if value > 10:
            print("\n", key, "为职业打假人/企业。")
# 关闭文件
f.close()





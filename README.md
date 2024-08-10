**目录**
一、数据准备	1
1.Doccano介绍	1
2.Uie介绍	2
3.模型部署步骤：	2
二、数据处理	3
三、预训练模型准备	5
1.环境准备	5
2.模型导出	6
四、模型代码	7
1.doccano.py:	8
2.finetune.py:	13
3.utils.py:	20
五、模型训练	40
1.Uie模型微调	40
2.运行截图：	41
六、实现效果	45
1.test.py代码:	45
2.运行截图：	48
七、报错分析	50
1.最初数据标注启动失败	50
2.微调启动失败	51
3.数据集格式错误	52
4.微调参数错误	53
5.选错数据标注类型	56
6.运行模型未保存	57
7.运行一半模型自动停止运行	57



**一、数据准备 **
*1.Doccano介绍*
doccano是一个面向人类的开源文本注释工具。它为文本分类、序列标记和从序列到序列的任务提供了注释功能。可以为情绪分析、命名实体识别、文本摘要等创建标记数据。只要创建一个项目，上传数据，并开始进行注释，该工具可以在数小时内构建一个数据集。
 
*2.Uie介绍*
UIE(Universal Information Extraction)：Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。为了方便大家使用UIE的强大能力，PaddleNLP借鉴该论文的方法，基于ERNIE 3.0知识增强预训练模型，训练并开源了首个中文通用信息抽取模型UIE。该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。

UIE的优势
•	使用简单：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。实现开箱即用，并满足各类信息抽取需求。
•	降本增效：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，大幅度降低标注数据依赖，在降低成本的同时，还提升了效果。
•	效果领先：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

*3.模型部署步骤：*
1．	配置doccano数据标注环境
2．	进行数据标注
3．	导出数据
4．	将数据格式转换为prompt
5．	通过doccano.py进行数据集分批等操作
6．	配置模型运行环境
7．	下载uie-base模型
8．	测试测试样例
9．	修改参数微调模型
10．	成功保存微调模型
11．	编写实例代码，调用微调模型实现项目要求
**二、数据处理** 
利用doccano进行数据标注，通过docker容器运行doccano。

作为一次性设置，创建如下的Docker容器：
docker pull doccano/doccano

docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=password" \
  -v doccano-db:/data \
  -p 8000:8000 doccano/doccano
接下来，通过运行容器来启动doccano：
docker container start doccano
网页访问 http://127.0.0.1:8000/.
要停止容器，请执行以下操作：
docker container stop doccano -t 5
在容器中创建的所有数据都将在重新启动期间持续存在。
 

**三、预训练模型准备** 
以下是 UIE Python 端的部署流程，包括环境准备、模型导出和使用示例。

*1.环境准备*
UIE的部署分为 CPU 和 GPU 两种情况，请根据你的部署环境安装对应的依赖。
CPU端
CPU端的部署请使用如下命令安装所需依赖：
pip install -r deploy/python/requirements_cpu.txt
GPU端
为了在 GPU 上获得最佳的推理性能和稳定性，请先确保机器已正确安装 NVIDIA 相关驱动和基础软件，确保 CUDA >= 11.2，cuDNN >= 8.1.1，并使用以下命令安装所需依赖
pip install -r deploy/python/requirements_gpu.txt
如果有模型推理加速、内存显存占用优化的需求，并且 GPU 设备的 CUDA 计算能力 (CUDA Compute Capability) 大于等于 7.0，例如 V100、T4、A10、A100/GA100、Jetson AGX Xavier 等显卡，推荐使用半精度（FP16）部署。直接使用微调后导出的 FP32 模型，运行时设置 --use_fp16 即可。
如果 GPU 设备的 CUDA 计算能力较低，低于 7.0，只支持 FP32 部署，微调后导出模型直接部署即可。

*2.模型导出*
模型训练、压缩时已经自动进行了静态图的导出，保存路径${finetuned_model} 下应该有 *.pdimodel、*.pdiparams 模型文件可用于推理。
推理
CPU端推理样例
在CPU端，请使用如下命令进行部署
python deploy/python/infer_cpu.py --model_path_prefix ${finetuned_model}/model
部署UIE-M模型
python deploy/python/infer_cpu.py --model_path_prefix ${finetuned_model}/model --multilingual
可配置参数说明：
model_path_prefix: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为./export/model.pdiparams，则传入./export/model。
position_prob：模型对于span的起始位置/终止位置的结果概率 0~1 之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
max_seq_len: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
batch_size: 批处理大小，请结合机器情况进行调整，默认为 4。
multilingual：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型是多语言模型，需要设置为 True；默认为 False。

GPU端推理样例
在GPU端，请使用如下命令进行部署
python deploy/python/infer_gpu.py --model_path_prefix ${finetuned_model}/model --use_fp16 --device_id 0
部署UIE-M模型
python deploy/python/infer_gpu.py --model_path_prefix ${finetuned_model}/model --use_fp16 --device_id 0 --multilingual
可配置参数说明：
model_path_prefix: 用于推理的 Paddle 模型文件路径，需加上文件前缀名称。例如模型文件路径为./export/model.pdiparams，则传入./export/model。
use_fp16: FP32 模型是否使用 FP16 进行加速，使用 FP32、INT8 推理时不需要设置，默认关闭。
position_prob：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
max_seq_len: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
batch_size: 批处理大小，请结合机器情况进行调整，默认为 4。
device_id: GPU 设备 ID，默认为 0。
multilingual：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型是多语言模型，需要设置为 True；默认为 False。




**四、模型代码**
 
略

**五、模型训练**
*Uie模型微调*
将data文件下的doccano导出标注文件doccano.json转化到data文件中

使用下面的命令，使用 uie-base 作为预训练模型进行模型微调，将微调后的模型保存至./checkpoint/model_best
python finetune.py  --device cpu --logging_steps 10 --save_steps 10 --eval_steps 10 --seed 50 --model_name_or_path uie-base --output_dir ./checkpoint1/model_best --train_path data/train.txt --dev_path data/dev.txt --max_seq_length 512  --per_device_eval_batch_size 4 --per_device_train_batch_size  4 --num_train_epochs 20 --learning_rate 1e-5 --label_names 'start_positions' 'end_positions' --do_train --do_export --export_model_dir ./checkpoint1/model_best --overwrite_output_dir --disable_tqdm False --save_total_limit 2

GPU
python finetune.py      --device gpu     --logging_steps 10     --save_steps 10     --eval_steps 100     --seed 42     --model_name_or_path uie-base     --output_dir ./checkpoint/model_best     --train_path data/train.txt     --dev_path data/dev.txt      --max_seq_length 512      --per_device_eval_batch_size 16     --per_device_train_batch_size  16     --num_train_epochs 20     --learning_rate 1e-5     --label_names 'start_positions' 'end_positions'     --do_train     --do_eval     --do_export     --export_model_dir ./checkpoint/model_best     --overwrite_output_dir     --disable_tqdm True     --metric_for_best_model eval_f1     --load_best_model_at_end  True     --save_total_limit 1


python -u -m paddle.distributed.launch --gpus "1,2,3" finetune.py --device gpu --logging_steps 10 --save_steps 100 --eval_steps 100 --seed 42 --model_name_or_path uie-m-large --output_dir ./checkpoint/model_best --train_path data/train.txt --dev_path data/dev.txt  --max_seq_length 512  --per_device_eval_batch_size 16 --per_device_train_batch_size  16 --num_train_epochs 100 --learning_rate 1e-5 --do_train --do_export --export_model_dir ./checkpoint/model_best --label_names 'start_positions' 'end_positions'--overwrite_output_dir --disable_tqdm True --metric_for_best_model eval_f1 --load_best_model_at_end  True --save_total_limit 2 –multilingual True


**六、实现效果**
*1.test.py代码:*
# 单个案例分析
from pprint import pprint
from paddlenlp import Taskflow

# 定义schema，分开原告和被告的身份信息
schema = ['原告-个人', '原告-企业', '被告-个人', '被告-企业']

# 创建信息抽取任务，加载指定的checkpoint
ie = Taskflow('information_extraction',
              checkpoint='D:/python/bert/数据法治/checkpoint1/model_best/checkpoint-60',
              schema=schema)

# 修改代码实现从文件中逐行获取文本并分析
# 打开文件
with open('D:/桌面/乐.txt', 'r', encoding='utf-8') as f:
    # 逐行读取
    for line in f:
        # 执行信息抽取
        results = ie(line)
        # 格式化输出结果
        pprint(results)
        # 只将results中'原告-企业':后面[]里面的内容写入乐out.txt文件中
        index = 0
        if isinstance(results, list) and len(results) > index:
            with open('D:/桌面/乐out.txt', 'a', encoding='utf-8') as f:
                f.write(str(results[index]) + '\n')
        else:
            print("Error: results 不是一个列表或索引超出范围")

# 关闭文件
f.close()

# 从乐out.txt文件中逐行读取文本，将原告-个人或者原告-企业中的原告姓名提取出来写入新建字典中，并统计原告姓名出现的次数
# 打开文件，并清空文件内容
with open('D:/桌面/乐out.txt', 'r', encoding='utf-8') as f:
    # 新建字典
    name_dict = {}
    # 逐行读取
    for line in f:
        # 提取原告-个人或者原告-企业中的原告姓名，只提取'text':后面''里面的原告姓名
        if '原告-个人' in line or '原告-企业' in line:
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
            print("该人/业为职业打假人/企业:", key)
# 关闭文件
f.close()



**七、报错分析**
*1.最初数据标注启动失败*


解决方式：更换数据标注方式

*2.微调启动失败*


解决方式，询问有经验的同学，在同学帮助下成功启动
*3.数据集格式错误*


解决方式：多次尝试不同格式后，成功运行程序
*4.微调参数错误*
 
解决方式：失败多次，观阅uie官方文档、询问chatgpt并在同学帮助下成功运行
*5.选错数据标注类型*


解决方式：失败一次后重新选择类型标注，最终成功

*6.运行模型未保存*


解决方式：白白训练12小时后修改路径重新训练

*7.运行一半模型自动停止运行*


解决方式：参数设置有误，修改参数重新开始

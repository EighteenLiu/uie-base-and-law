import tkinter as tk
from tkinter import filedialog
from threading import Thread
from paddlenlp import Taskflow
from PIL import Image, ImageTk, ImageOps

# 定义schema，分开原告和被告的身份信息
schema = ['原告个人', '原告企业']

# 创建信息抽取任务，加载指定的checkpoint
ie = Taskflow('information_extraction',
              checkpoint='D:/python/bert/数据法治/checkpoint1/model_best/checkpoint-420',
              schema=schema)

# 定义分析函数
def analyze_file(file_path):
    name_dict = {}
    results_text = ""  # 用于保存分析结果文本
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                results = ie(line)
                print(results)
                index = 0
                if isinstance(results, list) and len(results) > index:
                    with open('D:/桌面/a2.txt', 'a', encoding='utf-8') as out_f:
                        out_f.write(str(results[index]) + '\n')
                else:
                    print("Error: results 不是一个列表或索引超出范围")

        with open('D:/桌面/a2.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if '原告个人' in line or '原告企业' in line:
                    name = line.split("'text': '")[1].split("'")[0]
                    if name in name_dict:
                        name_dict[name] += 1
                    else:
                        name_dict[name] = 1

        frequent_names = [key for key, value in name_dict.items() if value > 10]

        if frequent_names:
            results_text = "\n".join([f"{name} 为商业维权主体。" for name in frequent_names])
        else:
            results_text = "没有发现出现次数大于10的原告姓名。"
        #results_text += "\n通过大批量诉讼获得大量的赔偿额，实际上是通过诉讼方式谋取利益，这种牟利方式是应当被限制的（不正当），所以想通过对该主体获得赔偿额（也就是利益）征税来减少这种牟利模式。但同时强调不对普通商业维权主体征税，因为他们是正常受到损害的主体，需要赔偿额去填补损失。"

    except Exception as e:
        results_text = f"分析过程中出现错误: {e}"

    finally:
        # 更新结果文本框
        category_textbox.delete(1.0, tk.END)  # 清空之前的内容
        category_textbox.insert(tk.END, results_text)
        # 恢复按钮状态并隐藏“正在分析”标签
        analyze_btn.config(state=tk.NORMAL)
        analyzing_label.pack_forget()

# 定义选择文件的函数
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(tk.END, file_path)
        analyze_btn.config(state=tk.NORMAL)

# 启动分析
def start_analysis():
    file_path = file_entry.get()
    if file_path:
        analyze_btn.config(state=tk.DISABLED)  # 禁用按钮，防止重复点击
        analyzing_label.pack(pady=10)  # 显示“正在分析”标签
        # 启动新线程来处理分析任务
        analysis_thread = Thread(target=analyze_file, args=(file_path,))
        analysis_thread.start()

# 创建主窗口
root = tk.Tk()
root.title("商业维权分析系统")
root.geometry("1024x576")
root.resizable(True, True)

# 加载背景图片
bg_image_path = "D:/python/bert/数据法治/data/background.png"  # 替换为你的背景图片路径
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((1024, 576), Image.Resampling.LANCZOS)  # 调整图片大小以适应窗口
bg_photo = ImageTk.PhotoImage(bg_image)

# 使用Canvas设置背景图片
canvas = tk.Canvas(root, width=1024, height=576)
canvas.pack(fill='both', expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor='nw')

# 设置左侧输入框区域
left_frame = tk.Frame(canvas, bg='#e0f7fa', padx=20, pady=20)
canvas.create_window(15, 45, anchor='nw', window=left_frame)

# 添加标题
title_label = tk.Label(left_frame, text="案件输入与分析界面", bg='#e0f7fa', font=('Calibri', 18, 'bold'))
title_label.pack(anchor='w')

# 文件输入框
file_entry = tk.Entry(left_frame, width=40, font=('Calibri', 12))
file_entry.pack(pady=10)

# 选择文件按钮
file_btn = tk.Button(left_frame, text="选择文件", command=select_file, bg='#007acc', fg='white', font=('Calibri', 12, 'bold'))
file_btn.pack(pady=10)

# 开始分析按钮
analyze_btn = tk.Button(left_frame, text="案件分析", command=start_analysis, bg='#007acc', fg='white', font=('Calibri', 14, 'bold'))
analyze_btn.pack(pady=10)

# 分析进度提示
analyzing_label = tk.Label(left_frame, text="正在分析，请稍候...", bg='#e0f7fa', font=('Calibri', 12, 'italic'), fg='red')

# 设置右侧分析结果区域
right_frame = tk.Frame(canvas, bg='#ffffff', padx=20, pady=30)
canvas.create_window(350, 45, anchor='nw', window=right_frame)

# 分析结果标题
result_title_label = tk.Label(right_frame, text="案件分析结果展示区", bg='#ffffff', font=('Calibri', 18, 'bold'))
result_title_label.pack(anchor='w')

# 案件分类部分
category_frame = tk.Frame(right_frame, bg='#ffffff')
category_frame.pack(fill='x', pady=10)

category_label = tk.Label(category_frame, text="案件分类", bg='#ffffff', font=('Calibri', 14, 'bold'))
category_label.pack(anchor='w')

category_textbox = tk.Text(category_frame, wrap='word', height=5, bg='#f0f0f0', font=('Calibri', 12), bd=2, relief='solid')
category_textbox.pack(fill='x')

# 案件趋势部分 (图像占位符)
trend_frame = tk.Frame(right_frame, bg='#ffffff')
trend_frame.pack(fill='x', pady=10)

trend_label = tk.Label(trend_frame, text="案件趋势", bg='#ffffff', font=('Calibri', 14, 'bold'))
trend_label.pack(anchor='w')

# 添加图片占位符
trend_image_label = tk.Label(trend_frame, text="此处显示案件趋势图", bg='#f0f0f0', font=('Calibri', 12), width=60, height=10, relief='solid')
trend_image_label.pack(fill='x')

# 运行主循环
root.mainloop()

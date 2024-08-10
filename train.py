import paddle
import paddle.nn as nn
from paddlenlp.transformers import UIEModel, UIEDataset
# 定义超参数
batch_size = 32
learning_rate = 2e-5
epochs = 3
# 加载数据集
train_dataset = UIEDataset.from_file("data/train.txt")    # 在此处填写训练数据路径
dev_dataset = UIEDataset.from_file("data/dev.txt")        # 在此处填写验证数据路径
# 定义模型和优化器
model = UIEModel.from_pretrained('uclanlp/visualbert-vcr-ui-evqa')    # 加载预训练模型
optimizer = paddle.optimizer.AdamW(learning_rate=learning_rate, parameters=model.parameters())
# 定义训练循环
model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_data_loader):
        input_ids, token_type_ids, attention_mask, slot_mask, start_positions, end_positions = batch
        loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                     slot_mask=slot_mask, start_positions=start_positions, end_positions=end_positions)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    # 在每个epoch后评估模型
    model.eval()
    with paddle.no_grad():
        total_loss = 0
        for step, batch in enumerate(dev_data_loader):
            input_ids, token_type_ids, attention_mask, slot_mask, start_positions, end_positions = batch
            loss = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         slot_mask=slot_mask, start_positions=start_positions, end_positions=end_positions)
            total_loss += loss.numpy()[0]
        avg_loss = total_loss / len(dev_data_loader)
        print("Epoch {} dev loss: {}".format(epoch+1, avg_loss))
    model.train()
# 保存模型
paddle.save(model.state_dict(), "uie_model")    # 在此处填写模型保存路径
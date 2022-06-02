import torch
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc
def train(model, iterator, optimizer, criterion):
    # 返回值是一个两个元素的列表,分别是平均准确率和平均loss
    print("train begin")
    epoch_loss = 0  # 这一波训练的loss
    epoch_acc = 0
    model.train()
    # for batch in enumerate(iterator):
    for batch in iterator:
        # 调用lstm，输入text，返回预测值
        # 具体写法根据数据集处理后提供的接口而定
        article, _ = batch.article

        # print("label:",batch.label)
        pred = model(batch.article)

        # 计算acc和loss

        pred = torch.squeeze(pred.cuda(), 1)
        # m = nn.Softmax(dim=0)
        # pred=m(pred)
        # print("pred",pred)

        real_label = batch.label.float()
        # print("label:", real_label)

        loss = criterion(pred.cuda(), real_label.cuda())
        # print(loss.item())
        acc = binary_accuracy(pred.cuda(), real_label.cuda())
        # print(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # label  1*batch_size
        # pred batch_size*

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.article)
            label = batch.label.float()
            # print('eval_pred shape',predictions.shape)
            real_label = batch.label.float()
            predictions = torch.squeeze(predictions.cuda(), 1)

            loss = criterion(predictions.cuda(), real_label.cuda())

            acc = binary_accuracy(predictions.cuda(), real_label.cuda())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
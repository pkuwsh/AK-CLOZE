import torch
from main import model,dataset_test_1,test_iterator_1
def test(iterator):
    model.eval()
    for batch in iterator:
        #article,_=batch.article[0]
        #print(batch.article)
        pred=model(batch.article)
        #print(batch.article)
        pred=torch.squeeze(pred,1)
        #print(pred)
        #print(batch.label)
        t=torch.argmax(pred,0)
        if(t==0):
            #print(pred[t])
            print('A')
        elif(t==1):
            print('B')
        elif(t==2):
            print('C')
        elif(t==3):
            print('D')


def test_1(model,iterator):

    model.eval()
    right=0
    for batch in iterator:
        article=batch.article
        pred=model(article)
        pred=torch.squeeze(pred,1)
        t=torch.argmax(pred)
        label=batch.label
        #print(label)
        if(label[t]==1):
            #print(label[t])
            right+=1

    print(right/(dataset_test_1.__len__()/4))

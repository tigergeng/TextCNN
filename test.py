import torch
from config import parse_config
from data_loader import DataBatchIterator
from sklearn.metrics import f1_score,precision_score,recall_score
if __name__ == '__main__':
    config = parse_config()
    test_data = DataBatchIterator(config=config, is_train=False, dataset="test")
    test_data.load()
    model = torch.load('./results/model.pt')
    model.eval()
    data_iter=iter(test_data)
    count=0
    score_f1=0
    score_precision=0
    score_recall=0
    for idx,batch in enumerate(data_iter):
        model.zero_grad()
        truths=batch.label
        outputs=model(batch.sent)
        result=torch.max(outputs,1)[1]
        y_truth=truths.data.detach().numpy().tolist()
        y_pred=result.view(truths.size()).data.detach().numpy().tolist()
        score_f1+=f1_score(y_truth, y_pred, average='macro')
        score_precision+=precision_score(y_truth,y_pred,average='macro')
        score_recall+=recall_score(y_truth,y_pred,average='macro')
        count+=1
    size=8000
    score_f1=100.0*score_f1/count
    score_precision=100.0*score_precision/count
    score_recall=100.0*score_recall/count
    print("f1_score:{:.4f}%".format(score_f1))
    print("precision_score:{:.4f}%".format(score_precision))
    print("recall_score:{:.4f}%".format(score_recall))
    
import torch
import matplotlib.pyplot as plt

def to_list(tensor):
    return tensor.detach().cpu().tolist()


train_loss_set_ckpt = torch.load('/home/ubuntu/BERT_FineTuning_SQUAD2.0/SQUAD/BERTTraining/checkpoint-final/training_loss.pt')
train_loss_set = to_list(train_loss_set_ckpt)

plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

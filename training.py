import sys
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.dirname(sys.path[0]))

import torch.nn as nn
from torch.utils.data import DataLoader
from models.edge_global_tag_transformer_121_K4 import edge_global_tag_transformer_121_k4
from models.edge_hier_tag_transformer_121_K4 import edge_hier_tag_transformer_121_k4
from utils import *
# import ci_cython


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba'][0]]
modeling = [edge_global_tag_transformer_121_k4, edge_hier_tag_transformer_121_k4 ][0]
model_st = modeling.__name__
loss_fn = nn.MSELoss()



TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

LR = 0.0002
LOG_INTERVAL = 20
NUM_EPOCHS = 2000

def main():
    cuda_name = "cuda:0"
    if len(sys.argv) > 3:
        cuda_name = "cuda:" + str([3])
    print('cuda_name:', cuda_name)

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    for dataset in datasets:
        print('\nrunning on ', model_st + '_' + dataset)
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset + '_train')
            test_data = TestbedDataset(root='data', dataset=dataset + '_test')

            # make data PyTorch mini-batch processing ready
            # 读取训练数据这里，用了多线程去读取，数据放在内存中
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4,
                                      pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4,
                                     pin_memory=True)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            model_file_name = 'pretrained/model_' + model_st + '_' + dataset + '.model'
            result_file_name = 'pretrained/result_' + model_st + '_' + dataset + '.csv'
            for epoch in range(NUM_EPOCHS):
                time_s = time.time()
                train(model, device, train_loader, optimizer, epoch + 1)
                print('train cost time ', time.time() - time_s)
                time_s = time.time()
                G, P = predicting(model, device, test_loader)
                print('predict cost time ', time.time() - time_s)
                # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch, ci(G, P)]
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch]

                time_s = time.time()
                # ret.append(ci_cython.boost_ci(G, P))
                # 这里优化了她原来的ci函数
                ret.append(boost_ci_v3(G, P))
                print('ci cost time ', time.time() - time_s)

                # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]

                if ret[1] < best_mse:

                    torch.save(model.state_dict(), model_file_name)

                    best_epoch = epoch + 1
                    # 调用两次
                    # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch, ci(G, P)]
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret)))

                    best_mse = ret[1]
                    best_ci = ret[-1]
                    best_rm2 = ret[4]

                    print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
                          'best_rm2:', best_rm2,
                          model_st, dataset)
                    print('')
                else:
                    print('')
                    print('MSE:', ret[1], 'CI:', ret[-1], 'RM2:', ret[-3], 'No improvement since epoch ', best_epoch,
                          '; best_mse,best_ci:', best_mse, best_ci,
                          'best_rm2:', best_rm2,
                          model_st, dataset)
                    print('')

if __name__ == '__main__':
    main()
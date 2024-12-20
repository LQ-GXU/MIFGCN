import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
from engine import trainer
import os
from gongju import *
import pickle
import scipy.io as sio



def edge_index_func(matrix):
    # print("In edge index function")
    a, b = [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # if(matrix[i][j] == 1 and i != j):
            if(matrix[i][j] == 1):
                a.append(i)
                b.append(j)
    edge = [a,b]
    edge_index = torch.tensor(edge, dtype=torch.long)
    return edge_index

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str,
                    default='PEMS07', help='data path')
parser.add_argument('--adjdata', type=str,
                    default='data/la/adj_PEMS08.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str,
                    default='doubletransition', help='adj type')
parser.add_argument('--num_nodes', type=int,
                    default=358, help='number of nodes')
parser.add_argument('--channels', type=int,
                    default=16, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.0025, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay rat'
                                         'e')
parser.add_argument('--epochs', type=int, default=2000, help='')
parser.add_argument('--print_every', type=int, default=200, help='')
parser.add_argument('--save', type=str,
                    default='./logs/'+str(time.strftime('%Y-%m-%d-%H_%M_%S'))+"-", help='save path')

parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--es_patience', type=int, default=100,
                    help='quit if no improvement after this many iterations')

args = parser.parse_args()


def main():
    seed = 41
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    data = args.data





    if args.data == "PEMS08":
        args.data = "data/"+args.data
        args.num_nodes = 170
        args.adjdata = "data/PEMS08/adj_PEMS08.pkl"
        dtw_path = r'/your_path/dtw_PEMS08.csv'
        adj_path = r'/your_path/adj_08.npy'
        path = r'/your_path/PEMS08.npz'
        weather_path = r'/your_path/weather08.npz'





    device = torch.device(args.device)
    adj_mx = util.load_adj(
        args.adjdata, args.adjtype)



    dtw = np.genfromtxt(dtw_path, delimiter=',')
    adj = np.load(adj_path)

    dtw = edge_index_func(dtw)
    adj = edge_index_func(adj)

    num_of_weeks = 1
    num_of_days = 1
    num_of_hours =1
    num_for_predict = 12

    points_per_hour = 12
    merge = 0
    flow_feature=2
    weather_feature=6
    all_data = read_and_generate_dataset(path, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
                                         points_per_hour,flow_feature,merge)
    weather_data = read_and_generate_dataset(weather_path, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
                                         points_per_hour,weather_feature,merge)
    dataloader, indice= util.load_dataset(
              all_data,weather_data, args.batch_size, args.batch_size, args.batch_size,num_of_weeks,num_of_days,num_of_hours,points_per_hour)

    pre_adj = [torch.tensor(i).to(device) for i in adj_mx]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(args.num_nodes, num_for_predict, args.channels, args.dropout,
                     args.learning_rate, args.weight_decay, device, pre_adj,dataloader['mean'],
                     dataloader['std'], adj, dtw, num_of_weeks, num_of_days, num_of_hours,adj_mx)


    print("start training...", flush=True)

    for i in range(1, args.epochs+1):

        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()


        for iter, (train_recent, y) in enumerate(dataloader['train_loader'].get_iterator()):
            train_week = torch.Tensor(train_week.transpose(0,3,2,1)).to(device)
            train_day = torch.Tensor(train_day.transpose(0,3,2,1)).to(device)
            train_recent = torch.Tensor(train_recent.transpose(0,3,2,1)).to(device)
            trainy = torch.Tensor(y.transpose(0,3,2,1)).to(device)  # (32,1,170,12)
            metrics = engine.train(train_recent, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}'
                print(log.format(
                    iter, train_loss[-1],  train_rmse[-1], train_mape[-1]), flush=True)
        t2 = time.time()
        log = 'Epoch: {:03d}, Training Time: {:.4f} secs'
        print(log.format(i, (t2-t1)))
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (val_recent, y) in enumerate(dataloader['val_loader'].get_iterator()):
            val_week = torch.Tensor(val_week.transpose(0, 3, 2, 1)).to(device)
            val_day = torch.Tensor(val_day.transpose(0, 3, 2, 1)).to(device)
            val_recent = torch.Tensor(val_recent.transpose(0, 3, 2, 1)).to(device)
            valy = torch.Tensor(y.transpose(0, 3, 2, 1)).to(device)


            metrics = engine.eval(val_recent, valy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()

        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(train_loss=np.mean(train_loss), train_rmse=np.mean(train_rmse),
                       train_mape=np.mean(train_mape), valid_loss=np.mean(valid_loss),
                       valid_rmse=np.mean(valid_rmse), valid_mape=np.mean(valid_mape))
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape), flush=True)
        log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape), flush=True)

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i < 1:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(),
                           path+"best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 1:
                outputs = []
                targets = []
                realy = torch.Tensor(dataloader['y_test'].transpose(0, 3, 2, 1)).to(device)

                for iter, ( test_recent, y) in enumerate(dataloader['test_loader'].get_iterator()):
                    # test_week = torch.Tensor(test_week.transpose(0, 3, 2, 1)).to(device)
                    # test_day = torch.Tensor(test_day.transpose(0, 3, 2, 1)).to(device)
                    test_recent = torch.Tensor(test_recent.transpose(0, 3, 2, 1)).to(device)
                    testy = torch.Tensor(y.transpose(0, 3, 2, 1)).to(device)


                    with torch.no_grad():
                        dict_test = {'input3':test_recent}
                        output,dadj = engine.model(dict_test)
                        output=output.transpose(1,3)

                        preds = output*dataloader['std']+dataloader['mean']
                    outputs.append(preds.squeeze())
                    targets.append(testy.squeeze())

                yhat = torch.cat(outputs, dim=0)
                targetss = torch.cat(targets,dim=0)
                yhat = yhat[indice]
                targetss = targetss[indice]




                amae = []
                amape = []
                armse = []
                test_m = []

                for j in range(12):
                    pred = yhat[:, :, j]
                    real = targetss[:, :, j]
                    metrics = util.metric(pred, real)
                    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    print(log.format(j+1, metrics[0], metrics[2], metrics[1]))

                    test_m = dict(test_loss=np.mean(metrics[0]),
                                  test_rmse=np.mean(metrics[2]), test_mape=np.mean(metrics[1]))
                    test_m = pd.Series(test_m)

                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])

                log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(),
                               path+"best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:",
                          np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:",
                          mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(6).to_csv(f'{path}/train.csv')
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    # Output consumption
    print(
        "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid-1], 4)))

    engine.model.load_state_dict(torch.load(path+"best_model.pth"))
    outputs = []
    inputs = []
    realy = torch.Tensor(dataloader['y_test'].transpose(0, 3, 2, 1)).squeeze().to(device)

    for iter, (test_week, test_day, test_recent, y) in enumerate(dataloader['test_loader'].get_iterator()):
        test_week = torch.Tensor(test_week.transpose(0, 3, 2, 1)).to(device)
        test_day = torch.Tensor(test_day.transpose(0, 3, 2, 1)).to(device)
        test_recent = torch.Tensor(test_recent.transpose(0, 3, 2, 1)).to(device)
        testy = torch.Tensor(y.transpose(0, 3, 2, 1)).to(device)
        with torch.no_grad():
            dict_test = {'input1': test_week, 'input2': test_day, 'input3': test_recent}
            output = engine.model(dict_test).transpose(1, 3)
            preds = output * dataloader['std'] + dataloader['mean']
            # targets = testy * dataloader['std'] + dataloader['mean']
        outputs.append(preds.squeeze())
        inputs.append(targets.squeeze())

    yhat = torch.cat(outputs, dim=0)
    targetss = torch.cat(inputs,dim=0)

    yhat = yhat[:realy.size(0), ...]
    sio.savemat('sti_pems08.mat', {'prediction': yhat.cpu().numpy(), 'true': targetss.cpu().numpy()})
    amae = []
    amape = []
    armse = []
    test_m = []

    for i in range(12):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[2], metrics[1]))

        test_m = dict(test_loss=np.mean(metrics[0]),
                      test_rmse=np.mean(metrics[2]), test_mape=np.mean(metrics[1]))
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))

    test_m = dict(test_loss=np.mean(amae),
                  test_rmse=np.mean(armse), test_mape=np.mean(amape))
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(6).to_csv(f'{path}/test.csv')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))

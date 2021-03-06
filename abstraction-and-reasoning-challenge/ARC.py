#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""
import os
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tensorfromdata import AllTasksGroupedWithTest, data_openner
from descriptive_stats_new import TrainParameters
from tools import flattener
from multi_scale_networkconv import OutputNetwork


PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = Path(PATH)
TRAINING_PATH = DATA_PATH / 'training'
EVALUATION_PATH = DATA_PATH / 'evaluation'
TEST_PATH = DATA_PATH / 'test'
SUBMISSION_PATH = DATA_PATH / 'output'

"""list the tasks"""
# evaluation_tasks = sorted(os.listdir(EVALUATION_PATH))
# training_tasks = sorted(os.listdir(TRAINING_PATH))
testing_tasks = sorted(os.listdir(TEST_PATH))

"""extract the datas from tasks"""
# eval_tasks = data_openner(evaluation_tasks, EVALUATION_PATH)
# train_tasks = data_openner(training_tasks, TRAINING_PATH)
test_tasks = data_openner(testing_tasks, TEST_PATH)

"""transform datas for the dataloader"""
# eval_task_data = AllTasksGroupedWithTest(eval_tasks)
# train_task_data = AllTasksGroupedWithTest(train_tasks)
test_task_data = AllTasksGroupedWithTest(test_tasks)

"""parameters = dictionary of parameters for DataLoader and optim
(ex:dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)"""
parameters = dict(
    lr=0.001,
    batch_size=1,
    shuffle=False,
    epochs=1
    )


results = []
results100 = []
results200 = []
resultsmod = []
labels = []

"""Dataloader"""
data_loader = DataLoader(dataset=test_task_data, batch_size=parameters['batch_size'], shuffle=False)
t = 0

""" each bach == one task """
for batch in data_loader:
    print(f'task_{t}')
    t += 1
    test_torch, in_out, out_size, feats, dict_feats = batch
    print(out_size)
    outnet = OutputNetwork(feats, out_size=out_size)
    optimizer = optim.Adam(outnet.parameters(), lr=parameters['lr'])

    """learning epochs"""
    for epoch in range(parameters['epochs']):
        total_loss = 0
        n = 0
        for i in in_out:
            x, y = i
            out_mat = outnet(x, out_size, feats.float(), y)
            out_x, out_y = y[0].shape

            #calculate Loss
            loss = 0
            loss = F.cross_entropy(out_mat.view((out_x*out_y), 10), y.view(-1))
            optimizer.zero_grad()
            #BackProp
            loss.backward()
            #update weights
            optimizer.step()

            total_loss += loss.item()
        print("epoch:", epoch, "/  Loss:", loss, "/  Total Loss:", total_loss)
        n += 1

        if n == 50:
            for test in test_torch:
                test_mat = test[0]
                test_id = test[1][0]
                pred_mat = outnet(test_mat, out_size, feats.float())
                labels.append(test_id)
                results100.append(flattener(pred_mat.argmax(dim=3).tolist()[0]))

        if n == 300:
            for test in test_torch:
                test_mat = test[0]
                pred_mat = outnet(test_mat, out_size, feats.float())
                pred_str = flattener(pred_mat.argmax(dim=3).tolist()[0])
                results200.append(pred_str)

                testfeat = dict()
                testfeat['input'] = test_mat[0][0][0].int().tolist()
                testfeat['output'] = pred_mat.argmax(dim=3)[0].tolist()

                tested = TrainParameters(testfeat,0)
                tested.basic_params()
                tested.colors_params()
                tested.colors_in_out()
                tested.others()

"""transform results for submission"""
for idx, ele in enumerate(results100):
    results.append(f'{results100[idx]} {results200[idx]} {resultsmod[idx]}')

SUBMISSION_PATH = Path('submission.csv')
results_dict = {'output_id': labels, 'output' : results}
res = pd.DataFrame(results_dict, columns=['output_id', 'output'])
res.to_csv(SUBMISSION_PATH, index=False)
    
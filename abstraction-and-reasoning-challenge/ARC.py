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
from multi_scale_networkconv import OutputNetwork
import torch.optim as optim
import torch.nn.functional as F
from tensorfromdata import AllTasksGroupedWithTest,  data_openner
from plot_task import plot_pred

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = Path(PATH)
TRAINING_PATH = DATA_PATH / 'training'
EVALUATION_PATH = DATA_PATH / 'evaluation'
TEST_PATH = DATA_PATH / 'test'
SUBMISSION_PATH = DATA_PATH / 'output'

# evaluation_tasks = sorted(os.listdir(EVALUATION_PATH))
# training_tasks = sorted(os.listdir(TRAINING_PATH))
testing_tasks = sorted(os.listdir(TEST_PATH))

# eval_tasks = data_openner(evaluation_tasks, EVALUATION_PATH)
# train_tasks = data_openner(training_tasks, TRAINING_PATH)
test_tasks = data_openner(testing_tasks, TEST_PATH)

# eval_task_data = AllTasksGroupedWithTest(eval_tasks)
# train_task_data = AllTasksGroupedWithTest(train_tasks)
test_task_data = AllTasksGroupedWithTest(test_tasks)

#parameters = dictionary of parameters for DataLoader and optim (dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
parameters = dict(
    lr=0.001,
    batch_size=1,
    shuffle=False,
    epochs=1
    )
results = []
labels = []
data_loader = DataLoader(dataset=test_task_data, batch_size=parameters['batch_size'], shuffle=False)
t=0
for batch in data_loader:
    print(f'task_{t}')
    t+=1
    test_torch, in_out, out_size, feats, dict_feats = batch
    print(out_size)
    outnet= OutputNetwork(feats, out_size=out_size)    
    optimizer = optim.Adam(outnet.parameters(), lr=parameters['lr'])
    for epoch in range(parameters['epochs']):
        total_loss = 0
        n = 0
        for i in in_out:
            x, y = i
            out_mat = outnet(x, out_size, feats.float(),y)
            out_x, out_y = y[0].shape

            #calculate Loss
            loss = 0
            loss = F.cross_entropy(out_mat.view((out_x*out_y),10), y.view(-1))
            optimizer.zero_grad()
            #BackProp
            loss.backward()
            #update weights
            optimizer.step()
            n += 1
            total_loss += loss.item()
        print("epoch:", epoch, "/  Loss:", loss, "/  Total Loss:", total_loss)
    
    for test in test_torch:
        test_mat = test[0]
        test_id = test[1][0]
        print(test_id)
        pred_mat = outnet(test_mat, out_size, feats.float())
        labels.append(test_id)
        results.append(flattener(pred_mat.argmax(dim=3).tolist()[0]))
        print(pred_mat.argmax(dim=3).shape)
        plot_pred(pred_mat.argmax(dim=3))
    
SUBMISSION_PATH = Path('submission.csv')
results_dict = {'output_id': labels, 'output' : results}
res=pd.DataFrame(results_dict, columns=['output_id', 'output'])
res.to_csv(SUBMISSION_PATH, index=False, header=False)
    
    
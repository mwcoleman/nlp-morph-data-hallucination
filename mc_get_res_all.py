import pandas as pd
import numpy as np
import re
import os, sys
import pprint


train_loss = re.compile(r'(loss is )(\d+.\d+)( at epoch )(\d+)')
dev_loss = re.compile(r'(dev loss is )(\d+.\d+)( at epoch )(\d+)')

dev_acc = re.compile(r'(dev accuracy is )(\d+.\d+)( at epoch )(\d+)')
dev_edit = re.compile(r'(dev average edit distance is )(\d+.\d+)( at epoch )(\d+)')

test_acc_re = re.compile(r'(acc )(\d+.\d+)')

init = list(zip([i+1 for i in range(400)], [0 for i in range(400)], [0 for i in range(400)]))

df = pd.DataFrame(init, columns=['epoch', 'train_loss', 'dev_loss'])

train_d = dict()
dev_d = dict()
dev_d_acc = dict()
dev_d_edit = dict()

os.chdir('/home/matt/fit5217/neural-transducer/results/original')
files = os.listdir(os.getcwd())


langs = ['izh', 'mwf', 'mlt']

# for i in range(len(langs),-1):
#     if langs[i] not in files:
#         langs.pop(i)

df_perf = pd.DataFrame(np.zeros((len(sys.argv)-1, len(langs)*4)),columns=['IZH_Dev_Acc',	'IZH_Dev_Edit',
                                'ZH_test_Acc','IZH_test_Edit','MWF_Dev_Acc',
                                'MWF_Dev_Edit','MWF_test_Acc','MWF_test_Edit',
                                'MLT_dev_Acc','MLT_dev_Edit','MLT_test_Acc','MLT_test_Edit'] )

for v in range(1,len(sys.argv)):

    perf = []

    for lang in langs:
        train_d = dict()
        dev_d = dict()
        dev_d_acc = dict()
        dev_d_edit = dict()
        
        with open('/home/matt/fit5217/neural-transducer/results/original/'+lang+sys.argv[v]+'.log') as f:
            l = f.readlines()
            for line in l:
                
                res_train = train_loss.search(line)
                res_dev = dev_loss.search(line)

                res_dev_acc = dev_acc.search(line)
                res_dev_edit = dev_edit.search(line)

                try:
                    t_loss, t_epoch = res_train.group(2), res_train.group(4)
                    train_d[t_epoch] = t_loss

                except:
                    pass
                try:
                    d_loss, d_epoch = res_dev.group(2), res_dev.group(4)
                    dev_d[int(d_epoch)] = d_loss
                except:
                    pass
                
                try:
                    d_acc, d_acc_epoch = res_dev_acc.group(2), res_dev_acc.group(4)
                    dev_d_acc[int(d_acc_epoch)] = d_acc
                except:
                    pass
                try:
                    d_edit, d_edit_epoch = res_dev_edit.group(2), res_dev_edit.group(4)
                    dev_d_edit[int(d_edit_epoch)] = d_edit
                except:
                    pass  

            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(dev_d_acc)
            # Now print out the test acc / edit distance
            max_dev_acc = max([float(v) for v in dev_d_acc.values()])
            max_dev_epoch = [k for k, v in dev_d_acc.items() if float(v) == max_dev_acc][0]
            min_dev_edit = dev_d_edit[max_dev_epoch]

            print(f"Extracted epoch.vs error for {lang+sys.argv[1]}. Best Dev result: \
                \nAcc: {max_dev_acc} -- Edit: {min_dev_edit} -- Epoch: {max_dev_epoch}\n---\nTest results are:\n{l[-1]}\n-----------")
            
            test_acc = test_acc_re.search(l[-1]).group(2)
            test_edit = l[-1][-7:-1]
            # Concat language perf together into one row. Each version gets a row.. 12 entries
            perf = perf + [max_dev_acc, min_dev_edit, test_acc, test_edit]
    # perf = [float(p) for p in perf]
    df_perf.loc[v-1] = perf
    print(df_perf)

    df_train = pd.DataFrame(train_d.items(), columns=['epoch','train_loss'])
    df_dev = pd.DataFrame(dev_d.items(), columns=['epoch','dev_loss'])            
    
    os.chdir('/home/matt/fit5217/neural-transducer/')
    df_train.to_csv('results/csv/'+lang+sys.argv[1]+'_train_loss.csv',sep=',')
    df_dev.to_csv('results/csv/'+lang+sys.argv[1]+'_dev_loss.csv',sep=',')

# df_perf = pd.DataFrame(df_perf, columns=['IZH_Dev_Acc',	'IZH_Dev_Edit',
#                                 'ZH_test_Acc','IZH_test_Edit','MWF_Dev_Acc',
#                                 'MWF_Dev_Edit','MWF_test_Acc','MWF_test_Edit',
#                                 'MLT_dev_Acc','MLT_dev_Edit','MLT_test_Acc','MLT_test_Edit'])
df_perf.to_csv('results/custom_csv/batch_results.csv',sep=',')         
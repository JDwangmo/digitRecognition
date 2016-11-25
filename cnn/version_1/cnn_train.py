#encoding=utf8
from dataProcessing.data_util import DataUtil
from dataProcessing.read_data import load_pix
import logging
import timeit
import pandas as pd
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG
                    )
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
config = yaml.load(file('./config.yaml'))
config = config['cnn_train']
verbose = config['verbose']
logging.basicConfig(filename=''.join(config['log_file_path']),filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print(config['describe'])
print('='*30)
print('start running!')
logging.debug('='*30)
logging.debug(config['describe'])
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)


# logging.debug( 'the shape of train sample:%d,%d,%d,%d'%(train_pix.shape))
# logging.debug( 'the shape of test sample:%d,%d,%d,%d'%(test_pix.shape))
# quit()

dutil = DataUtil(charset_type=config['charset_type'])
num_labels = len(dutil.index_to_character)

logging.debug('charset type 为：%s,字符数：%d'%(str(config['charset_type']),num_labels))
print('charset type 为：%s,字符数：%d'%(str(config['charset_type']),num_labels))


from cnn_model.image_net_model import AlexNet
counter = 0

for (train_X,train_y),(test_X,test_y) in dutil.get_train_test():

    net = AlexNet(verbose = config['verbose'],
                  num_labels=num_labels,
                  nb_epoch=config['nb_epoch']
                  )
    net.print_model_descibe()

    net.fit((train_X,train_y),(test_X,test_y))
    _, _, _, _,is_correct, y_pred = net.accuracy((test_X,test_y))

    # print(len(X_train))
    pd.DataFrame(data={'LABEL':test_y,'PREDICT':y_pred,'CORRECT':is_correct}).to_csv(
        'result/result%d_%depoch_%stype.csv'%(counter,config['nb_epoch'],config['charset_type']),
        sep='\t',
        encoding='utf8',
        index=False,
    )
    quit()
    counter += 1









end_time = timeit.default_timer()
print('end! Running time:%ds!'%(end_time-start_time))
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))
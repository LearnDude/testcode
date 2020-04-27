import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import keras.backend as K

from keras import layers, regularizers
from keras.layers import Input, Dense 
from keras.models import Model
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.utils import multi_gpu_model, Sequence
# from keras.applications.imagenet_utils import preprocess_input
# import keras
# from keras.utils.vis_utils import model_to_dot
# from keras.losses import mse, binary_crossentropy
# from keras.utils import plot_model



### Constants
ROOTPATH = './'
SRCPATH = ROOTPATH + 'src/'
DATAPATH = ROOTPATH + 'data/'
OUTPUTPATH = ROOTPATH + 'output/'
TRAINPATH = DATAPATH + 'train.csv'
TESTPATH = DATAPATH + 'test.csv'
SUBMISSIONPATH = DATAPATH + 'submission_split.csv'

### Code
def main():

    # Read data
    train_df = pd.read_csv(TRAINPATH)
    test_df = pd.read_csv(TESTPATH)
    all_df = [train_df, test_df]

    # Examine & Clean data
    for df in all_df:

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

        df['Sex'].replace(to_replace = ['male', 'female'], value = [0,1], inplace = True)
        df['Age'].fillna( df['Age'].mean(), inplace = True)
        df['Fare'].fillna( df['Fare'].mean(), inplace = True)
        df['Embarked'].fillna( 'S', inplace = True)
        df['Embarked'].replace(to_replace = ['S', 'C', 'Q'], value = [-1, 0, 1], inplace = True)

        # print(df.columns)
        # print(df.count())
        # print(df.describe())
        
        # for col in df: 
        #     print(col, df[col].unique())
        
        print(df.head)


    train_x = train_df.to_numpy()[:,1:]
    train_y = train_df.to_numpy()[:,0]
    test_x = test_df.to_numpy()
    print(train_x.shape, test_x.shape)

    train_dim = train_x.shape[1:]

    L_input = Input(shape = train_dim)

    L_dense1 = Dense(128, activation = 'sigmoid')(L_input)
    L_dense2 = Dense(128, activation = 'sigmoid')(L_dense1)
    L_dense3 = Dense(128, activation = 'sigmoid')(L_dense2)

    L_output = Dense(1, activation = 'sigmoid' )(L_dense3)

    log_reg = Model(inputs = L_input, outputs = L_output, name = 'Log reg model')

    log_reg.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

    log_reg.fit(x = train_x, y = train_y, epochs = 500)

    

    # best_acc = 0
    best_st = 0.5
    # for st in range(200):
    #     y_tune = log_reg.predict(x = train_x)
    #     y_tune = ( y_tune > float(st/200))
    #     tune_acc = log_reg.evaluate(x = train_x, y = y_tune, verbose = 0 )[1]
        
    #     if tune_acc > best_acc:
    #         best_st = st/200
    #         best_acc = tune_acc
    #         print('best threshold', best_st, best_acc)
    
    train_eval = log_reg.evaluate(x = train_x, y = (train_y > best_st) )
    print("train eval", train_eval, log_reg.metrics_names)

    test_y = log_reg.predict(x = test_x)
    
    test_y = ( test_y > best_st )


    sub_df = pd.read_csv(SUBMISSIONPATH)
    sub_df['Survived'] = test_y.astype(int)
    sub_df.to_csv(OUTPUTPATH + 'submission.csv', index = False)














if __name__ == '__main__': main()
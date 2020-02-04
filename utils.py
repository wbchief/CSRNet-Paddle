import paddle.fluid as fluid
try:
    from CSRNet import CSRNet
    from config import train_parameters
    from dataloader import SH_data_loader
except:
    from work.CSRNet import CSRNet
    from work.config import train_parameters
    from work.dataloader import SH_data_loader
from tqdm import tqdm
import numpy as np
import os
import cv2 


def mse_loss(predicts, labels):
    square_loss = fluid.layers.square_error_cost(predicts, labels)
    
    loss = fluid.layers.reduce_sum(square_loss, dim=[1,2,3])
    #print(loss.shape)
    loss = 0.5 * fluid.layers.mean(loss)
    return loss


    
    
def eval1(model_path, test_reader, method):
    
    #method = train_parameters['method']
    #model_path = ''
    with fluid.dygraph.guard():
        
		print("CSR")
		net = CSRNet("CSR")    
        model_dict, _ = fluid.load_dygraph(model_path)
        net.load_dict(model_dict)
        net.eval()
        print('start eval!')
  
        mae=0
        mse = 0
        val_loss = 0
        for batch_id, data in enumerate(test_reader()):
            image = np.array([x[0] for x in data]).astype('float32')
            label = np.array([x[1] for x in data]).astype('float32')
            
            image = fluid.dygraph.to_variable(image)
            label = fluid.dygraph.to_variable(label)
            label.stop_gradient = True
            predict = net(image)
            loss = mse_loss(predict, label)
            val_loss += loss
            mae+=abs(predict.numpy().sum()-label.numpy().sum())
            mse += (predict.numpy().sum()-label.numpy().sum())*(predict.numpy().sum()-label.numpy().sum())
            if batch_id % 99 ==0:
                print(batch_id, 'predict:', predict.numpy().sum(), 'real:', label.numpy().sum())
        
        print('counts:', batch_id+1, 'loss:',val_loss.numpy()[0], 'avg_loss', val_loss.numpy()[0] / (batch_id+1), "mae:", str(mae/(batch_id+1)), 'mse:', mse/(batch_id+1))
        print('real:', label.numpy().sum(), 'predict:', predict.numpy().sum())




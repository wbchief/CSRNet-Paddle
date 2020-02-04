import paddle.fluid as fluid
import numpy as np
import paddle
try:

    from config import train_parameters, init_train_parameters
    from dataloader import SH_data_loader
    from utils import mse_loss
    from CSRNet import CSRNet
except:
 
    from work.config import train_parameters, init_train_parameters
    from work.dataloader import SH_data_loader
    from work.utils import mse_loss
    from work.CSRNet import CSRNet
import math
import os




def optimizer_setting(params):
    momentum_rate = 0.95
    l2_decay = 1.2e-4
    ls = params["learning_strategy"]
    if "image_count" not in params:
        image_count = 400
    else:
        image_count = params["image_count"]

    batch_size = ls["batch_size"]
    step = int(math.ceil(float(image_count) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.cosine_decay(learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
       
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay)
        )

    return optimizer
    
def train():
    method = train_parameters['method']
    print(method)
    save_dir = train_parameters['save_dir']
    print(save_dir)

    
    train_reader = paddle.batch(SH_data_loader('/home/aistudio/sh/sh/part_B_final/train_data/images/', size=[256, 512], mode='train', scale=8),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=False)
    test_reader = paddle.batch(SH_data_loader('/home/aistudio/sh/sh/part_B_final/test_data/images/', size=[256, 512],mode='val', scale=8),
                                batch_size=1,
                                drop_last=False)
    
    with fluid.dygraph.guard():
        epoch_num = train_parameters["num_epochs"] # 5
        print("epocj_num", epoch_num)
        
       
		print("CSR")
		net = CSRNet("CSR")
            
       
        
        print('train')
        optimizer = optimizer_setting(train_parameters)
        #optimizer = fluid.optimizer.SGD(1e-6,momentum=0.95)
        
        if train_parameters["continue_train"]:
            # 加载上一次训练的模型，继续训练
            
            model, _ = fluid.load_dygraph(train_parameters['continue_train_dir'])
            net.load_dict(model)
            optimizer.set_dict(_)
            print('继续训练', train_parameters['continue_train_dir'])
        
        best_mae = 1000000
        min_epoch=0
        for epoch in range(epoch_num):
          
            epoch_loss = 0
            #mae = 0
            for batch_id, data in enumerate(train_reader()):
                image = np.array([x[0] for x in data]).astype('float32')
                label = np.array([x[1] for x in data]).astype('float32')
        
                image = fluid.dygraph.to_variable(image)
                label = fluid.dygraph.to_variable(label)
                label.stop_gradient = True
                predict = net(image)
                loss = mse_loss(predict, label)
                backward_strategy = fluid.dygraph.BackwardStrategy()
                backward_strategy.sort_sum_gradient = True
                loss.backward(backward_strategy)
                epoch_loss+=loss.numpy()[0]
                #print(net._x_for_debug.gradient())
                optimizer.minimize(loss)
                net.clear_gradients()
                #mae+=abs(predict.numpy().sum()-label.numpy().sum())
            print('epoch:', epoch, 'loss:', epoch_loss)
                
            # dy_param_value = {}
            # for param in net.parameters():
            #     dy_param_value[param.name] = param.numpy()
           
            # fluid.save_dygraph(net.state_dict(), save_dir + method + str(epoch))
            # fluid.save_dygraph(optimizer.state_dict(), save_dir + method + str(epoch))
            
            net.eval()
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
                val_loss += loss.numpy()[0]
                mae += abs(predict.numpy().sum()-label.numpy().sum())
                mse += (predict.numpy().sum()-label.numpy().sum())*(predict.numpy().sum()-label.numpy().sum())
            net.train()    
            if mae/(batch_id+1)<best_mae:
                best_mae=mae/(batch_id+1)
                min_epoch=epoch
                fluid.save_dygraph(net.state_dict(), save_dir + method + str(epoch))
                fluid.save_dygraph(optimizer.state_dict(), save_dir + method + str(epoch))
            print("test epoch:", str(epoch), 'loss:',val_loss, " error:", str(mae/(batch_id+1)), " min_mae:", str(best_mae), " min_epoch:", str(min_epoch), 
                    'mse:', mse/(batch_id+1), 'real:', label.numpy()[0].sum(), 'pre:', predict.numpy()[0].sum())
            del mae, mse, image, label, predict
            
if __name__ == '__main__':
    init_train_parameters()
    train()


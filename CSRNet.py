

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import pickle

import pickle
with open('/home/aistudio/work/vgg16.pkl', 'rb') as f:
    mode = pickle.load(f)
#print(mode.keys())
# 【0， 2， 5， 7， 10， 12， 14， 17， 19， 21】
# features.0.weight
class VGG16(fluid.dygraph.Layer):
    def __init__(self, name_scope, cfg, batch_norm=False, dilation=False):
        super(VGG16, self).__init__(name_scope)
        self.cfg = cfg
        self.batch_norm = batch_norm
        self.dilation = dilation
        self.layers = self.make_layers(self.cfg)
        #print(self.layers[0])
    
    def forward(self, x):
        for op in self.layers:
            x = op(x)
        return x
    
    def make_layers(self, cfg, batch_norm=False, dilation=False):
        '''
        cfg: 参数
        batch_norm: 是否正则化
        dilation: 膨化系数， frontend:1, backend:2
        
        '''
        
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        i = 0
        for v in cfg:
            if v == 'M':
                layers.append(self.add_sublayer('pool_' + str(i), Pool2D(self.full_name(), pool_size=2, pool_stride=2)))
                
            else:
                #layers.append(Conv2D(num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate))
                if batch_norm:
                    layers.append(self.add_sublayer('conv_' + str(i), Conv2D(self.full_name(), num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate,param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.NormalInitializer(scale=0.01) ), bias_attr=fluid.ParamAttr())))
                    layers.append(self.add_sublayer('batch_norm_' + str(i), BatchNorm(v, act="relu",param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.ConstantInitializer(value=1) ), bias_attr=fluid.ParamAttr(
                                                initializer=fluid.initializer.ConstantInitializer(value=0)))))
                else:
                    layers.append(self.add_sublayer('conv_' + str(i), Conv2D(self.full_name(), num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate, act='relu', param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.NormalInitializer(scale=0.01) ), bias_attr=fluid.ParamAttr())))
            i+=1
        return layers
class FrontEnd(fluid.dygraph.Layer):
    def __init__(self, name_scope, cfg, batch_norm=False, dilation=False):
        super(FrontEnd, self).__init__(name_scope)
        # 【0， 2， 5， 7， 10， 12， 14， 17， 19， 21】
        self.name = ['features.0', 'features.2', 'M', 'features.5', 'features.7', 'M', 'features.10', 'features.12', 'features.14', 'M', 'features.17', 'features.19', 'features.21']
        self.layers = self.make_layers(cfg, self.name)
      
    def forward(self, x):
       
        for op in self.layers:
            x = op(x)
            #print(op.bias)
        return x
    
    def make_layers(self, cfg, name, batch_norm=False, dilation=False):
        '''
        cfg: 参数
        batch_norm: 是否正则化
        dilation: 膨化系数， frontend:1, backend:2
        
        '''
        
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v, n in zip(cfg, name):
            if v == 'M':
                layers.append(self.add_sublayer('pool_' + str(n), Pool2D(self.full_name(), pool_size=2, pool_stride=2)))
            else:
                #layers.append(Conv2D(num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate))
                if batch_norm:
                    layers.append(self.add_sublayer('conv_' + str(n), Conv2D(self.full_name(), num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate,param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.NumpyArrayInitializer(mode[n + '.weight'])), bias_attr=fluid.ParamAttr(
                                                initializer=fluid.initializer.NumpyArrayInitializer(mode[n + '.bias'])) )))
                    layers.append(self.add_sublayer('batchnorm_' + str(n), BatchNorm(v, act="relu",param_attr=fluid.ParamAttr(
                                                initializer=fluid.initializer.ConstantInitializer(value=1) ), bias_attr=fluid.ParamAttr(
                                                initializer=fluid.initializer.ConstantInitializer(value=0)))))
                else:
                    layers.append(self.add_sublayer('conv_' + str(n), Conv2D(self.full_name(), num_filters=v, filter_size=3, padding=d_rate, dilation=d_rate, act='relu', param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.NumpyArrayInitializer(mode[n + '.weight'])),bias_attr=fluid.ParamAttr(
                                                    initializer=fluid.initializer.NumpyArrayInitializer(mode[n + '.bias'])) )))
            
        return layers
    
class CSRNet(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(CSRNet, self).__init__(name_scope)
        # 'M' 池化
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat=[512, 512, 512, 256, 128, 64]
        self.frontend = self.add_sublayer('VGG16', FrontEnd('frontend', self.frontend_feat))
       
        self.backend = self.add_sublayer('backend', VGG16('backend', self.backend_feat, batch_norm=True, dilation=True))
        self.output_layer = Conv2D(self.full_name(), num_filters=1, filter_size=1, param_attr=fluid.ParamAttr(
                                            initializer=fluid.initializer.NormalInitializer(scale=0.01) ), bias_attr=fluid.ParamAttr(
                                                initializer=fluid.initializer.ConstantInitializer(value=0)))
        
    def forward(self, x):
       
        x = self.frontend(x)  # NCHW
        
        x = self.backend(x)
        x = self.output_layer(x)
   
        return x
        
'''

'''       

if __name__ == '__main__':
    with fluid.dygraph.guard():
        net = CSRNet('csrnet')
        img = np.random.rand(1, 3, 256, 512).astype('float32')
        label = np.zeros([1, 1, 64, 128]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        label = fluid.dygraph.to_variable(label)
        outs = net(img)
        print(net.state_dict().keys())
        print(len(list(net.state_dict().keys())))
        #outs.backward()
        #print(net.state_dict().keys())
        #loss = mse_loss(outs, label)
        #mean_loss = fluid.layers.mean(loss)
        #loss.backward()
        #print(outs.numpy())
        #print(loss)
        #print("acc", evaluate(outs, label))
            
    
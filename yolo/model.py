import tensorflow as tf
import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] ='2'



class CreateConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self,filters,stride=2,kern_size=(3,3)):
        super(CreateConvolutionLayer,self).__init__()
        self.conv_1= tf.keras.layers.Conv2D(filters,kern_size, stride,activation=tf.nn.leaky_relu, padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()

    def call(self,input, training= False):
        x= self.conv_1(input)
        return self.bn_1(x)


class CreateResidualSection(tf.keras.layers.Layer):
    def __init__(self,filter1,filter2):
        super(CreateResidualSection,self).__init__()
        self.c1= CreateConvolutionLayer(filter1,stride=1,kern_size=(1,1))
        self.c2=  CreateConvolutionLayer(filter2,stride=1)

    def call(self,input):
        x=self.c1(input)
        x= self.c2(x)
        return tf.keras.layers.add([input,x])



class Darknet(tf.keras.Model):
    def __init__(self):
        super(Darknet,self).__init__()
        self.co1= CreateConvolutionLayer(32,stride=1)
        self.co2= CreateConvolutionLayer(64)
        self.res1= CreateResidualSection(32,64)
        self.co3= CreateConvolutionLayer(128)
        self.res2 = [CreateResidualSection(64,128) for _ in range(2)]
        self.co4= CreateConvolutionLayer(256)
        self.res3= [CreateResidualSection(128,256) for _ in range(8)]
        self.co5= CreateConvolutionLayer(512)
        self.res4= [CreateResidualSection(256,512) for _ in range(8)]
        self.co6= CreateConvolutionLayer(1024)
        self.res5= [CreateResidualSection(512,1024) for _ in range(4)]
    
    def call(self,input):
        x = self.co1(input)
        x= self.co2(x)  
        x= self.res1(x)
        x= self.co3(x)
        for layer in self.res2:x= layer(x) #2
        x= self.co4(x)
        for layer in self.res3: x= layer(x) #8
        x1= self.co5(x)
        for layer in self.res4: x1= layer(x1) #8
        x2= self.co6(x1)
        for layer in self.res5: x2= layer(x2) #4

        return [x,x1,x2]


class Detection(tf.keras.layers.Layer):
    def __init__(self,n_channels):
        super(Detection,self).__init__()
        self.c1= [CreateConvolutionLayer(n_channels,stride=1,kern_size=(1,1)) for _ in range(3)]
        self.c2= [CreateConvolutionLayer(n_channels *2,stride=1) for _ in range(3)]

    def call(self,input):
        x= input
        for l1,l2 in zip(self.c1,self.c2):
            x= l1(x)
            x= l2(x)
        return x
 


class YOLO(tf.keras.Model):
    def __init__(self,nclass):
        super(YOLO,self).__init__()
        self.mdl = Darknet()
        self.det1= Detection(512)
        self.out1= CreateConvolutionLayer(3*(nclass + 5),stride=1,kern_size=(1,1))
        self.c1= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.up1= tf.keras.layers.UpSampling2D(2)
        self.det2= Detection(256)
        self.out2= CreateConvolutionLayer(3*(nclass + 5),stride=1,kern_size=(1,1))
        self.c2= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.up2= tf.keras.layers.UpSampling2D(2)
        self.det3= Detection(128)
        self.out3= CreateConvolutionLayer(3*(nclass + 5),stride=1,kern_size=(1,1))
    
    def call(self,input):
        x= self.mdl(input)
        x_ = self.det1(x[2])
        out1= self.out1(x_)
        x_ = self.c1(x_)
        x_ = self.up1(x_)
        x_= tf.keras.layers.concatenate([x[1],x_])
        x_ = self.det2(x_)
        out2= self.out2(x_)
        x_= self.c2(x_)
        x_ = self.up2(x_)
        x_= tf.keras.layers.concatenate([x[0],x_])
        x_= self.det3(x_)
        out3= self.out3(x_)
        
        print(out1.shape,out2.shape,out3.shape)
        return [out1, out2, out3]

    def model(self):
        x= tf.keras.Input(shape=(416,416,3))
        return tf.keras.Model(inputs=[x],outputs= self.call(x))        



if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='No of classes to be entered')
    parser.add_argument('n_classes',type=int)
    args= parser.parse_args()
    model= YOLO(nclass=args.n_classes)
    model.model().summary()
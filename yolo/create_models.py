import tensorflow as tf
import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] ='2'



class CreateConvolutionLayer(tf.keras.Model):
    def __init__(self,filters,stride=2,kern_size=(3,3)):
        super(CreateConvolutionLayer,self).__init__()
        self.conv_1= tf.keras.layers.Conv2D(filters,kern_size, stride,activation=tf.nn.leaky_relu, padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()

    def call(self,input, training= False):
        x= self.conv_1(input)
        return self.bn_1(x)


class CreateResidualSection(tf.keras.Model):
    def __init__(self,filter1,filter2):
        super(CreateResidualSection,self).__init__()
        self.c1= CreateConvolutionLayer(filter1,stride=1,kern_size=(1,1))
        self.c2=  CreateConvolutionLayer(filter2,stride=1)

    def call(self,input):
        x=self.c1(input)
        x= self.c2(x)
        return tf.keras.layers.add([input,x])

class CreateModel(tf.keras.Model):
    def __init__(self,nclasses):
        super(CreateModel,self).__init__()
        self.co1= CreateConvolutionLayer(32,stride=1)
        self.co2= CreateConvolutionLayer(64)
        self.res1= CreateResidualSection(32,64)
        self.co3= CreateConvolutionLayer(128)
        self.res2= CreateResidualSection(64,128)
        self.res3= CreateResidualSection(64,128)
        self.co4= CreateConvolutionLayer(256)
        self.res4= CreateResidualSection(128,256)
        self.res5= CreateResidualSection(128,256)
        self.res6= CreateResidualSection(128,256)
        self.res7= CreateResidualSection(128,256)
        self.res8= CreateResidualSection(128,256)
        self.res9= CreateResidualSection(128,256)
        self.res10= CreateResidualSection(128,256)
        self.res11= CreateResidualSection(128,256)
        self.co5= CreateConvolutionLayer(512)
        self.res12= CreateResidualSection(256,512)
        self.res13= CreateResidualSection(256,512)
        self.res14= CreateResidualSection(256,512)
        self.res15= CreateResidualSection(256,512)
        self.res16= CreateResidualSection(256,512)
        self.res17= CreateResidualSection(256,512)
        self.res18= CreateResidualSection(256,512)
        self.res19= CreateResidualSection(256,512)
        self.co6= CreateConvolutionLayer(1024)
        self.res20= CreateResidualSection(512,1024)
        self.res21= CreateResidualSection(512,1024)
        self.res22= CreateResidualSection(512,1024)
        self.res23= CreateResidualSection(512,1024)
        self.co7= CreateConvolutionLayer(512,stride=1,kern_size=(1,1))
        self.co8= CreateConvolutionLayer(1024,stride=1)
        self.co9= CreateConvolutionLayer(512,stride=1,kern_size=(1,1))
        self.co10= CreateConvolutionLayer(1024,stride=1)
        self.co11= CreateConvolutionLayer(512,stride=1,kern_size=(1,1))
        self.co12= CreateConvolutionLayer(1024,stride=1)
        self.det1= CreateConvolutionLayer(3*(nclasses + 5),stride=1,kern_size=(1,1))
        self.col13= CreateConvolutionLayer(256,stride=1,kern_size=(1,1))
        self.up1= tf.keras.layers.UpSampling2D(2)
        self.col14= CreateConvolutionLayer(256,stride=1,kern_size=(1,1))
        self.col15= CreateConvolutionLayer(512,stride=1)
        self.col16= CreateConvolutionLayer(256,stride=1,kern_size=(1,1))
        self.col17= CreateConvolutionLayer(512,stride=1)
        self.col18= CreateConvolutionLayer(256,stride=1,kern_size=(1,1))
        self.col19= CreateConvolutionLayer(512,stride=1)
        self.det2= CreateConvolutionLayer(3*(nclasses + 5),stride=1,kern_size=(1,1))
        self.col20= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.up2= tf.keras.layers.UpSampling2D(2)
        self.col21= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.col22= CreateConvolutionLayer(256,stride=1)
        self.col23= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.col24= CreateConvolutionLayer(256,stride=1)
        self.col25= CreateConvolutionLayer(128,stride=1,kern_size=(1,1))
        self.col26= CreateConvolutionLayer(256,stride=1)
        self.det3= CreateConvolutionLayer(3*(nclasses + 5),stride=1,kern_size=(1,1))




    def call(self,input):
        x= self.co1(input)
        x= self.co2(x)  
        x= self.res1(x)
        x= self.co3(x)
        x= self.res2(x)
        x= self.res3(x)
        x= self.co4(x)
        x= self.res4(x)
        x= self.res5(x)
        x= self.res6(x)
        x= self.res7(x)
        x= self.res8(x)
        x= self.res9(x)
        x= self.res10(x)
        x= self.res11(x)
        x1= self.co5(x)
        x1= self.res12(x1)
        x1= self.res13(x1)
        x1= self.res14(x1)
        x1= self.res15(x1)
        x1= self.res16(x1)
        x1= self.res17(x1)
        x1= self.res18(x1)
        x1= self.res19(x1)
        x2= self.co6(x1)
        x2= self.res20(x2)
        x2= self.res21(x2)
        x2= self.res22(x2)
        x2= self.res23(x2)
        x2= self.co7(x2)
        x2= self.co8(x2)
        x2= self.co9(x2)
        x2= self.co10(x2)
        x2= self.co11(x2)
        x2= self.co12(x2)
        #Detection for big objects
        det1= self.det1(x2)

        #Detection for medium objects
        x2= self.col13(x2)
        x2= self.up1(x2)
        x2 =tf.keras.layers.concatenate([x2,x1])
        x2= self.col14(x2)
        x2= self.col15(x2)
        x2= self.col16(x2)
        x2= self.col17(x2)
        x2= self.col18(x2)
        x2= self.col19(x2)
        det2= self.det2(x2)

        #Detection for small objects
        x2= self.col20(x2)
        x2= self.up1(x2)
        x2 =tf.keras.layers.concatenate([x2,x])
        x2= self.col21(x2)
        x2= self.col22(x2)
        x2= self.col23(x2)
        x2= self.col24(x2)
        x2= self.col25(x2)
        x2= self.col26(x2)
        det3= self.det3(x2)


        print(det1.shape, det2.shape,det3.shape)
        return [det1,det2,det3]

    def model(self):
        x= tf.keras.Input(shape=(416,416,3))
        return tf.keras.Model(inputs=[x],outputs= self.call(x))        



if __name__ == '__main__':
    parser= argparse.ArgumentParser(description='No of classes to be entered')
    parser.add_argument('noclasses',type=int)
    args= parser.parse_args()
    model= CreateModel(nclasses=args.noclasses)
    model.model().summary()
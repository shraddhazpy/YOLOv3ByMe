import tensorflow as tf
import os

def read_image(image_path):
    img= tf.io.read_file(image_path)
    img= tf.image.decode_image(img,channels=3)
    return img

def resize_image(image_path,size):
    img= tf.io.read_file(image_path)
    img= tf.image.decode_image(img,channels=3)
    img= tf.image.resize(img, (size, size))
    img= img/255
    return img

def parse_file(path):
    if os.path.isfile(path):
        img_wise_data=[]
        sub_list=[]
        with open(path,'r') as fl:
            text_file_parse = fl.readlines()
        text_file_parse= list(map(lambda x: x.rstrip('\n'),text_file_parse ))
        for i in text_file_parse:
            if i:
                sub_list.append(i)
            else:
                img_wise_data.append(sub_list)
                sub_list=[]  
        img_wise_data.append(sub_list)

        img_path_list= list(map(lambda x: x[0],img_wise_data[:-1]))
        y_true = list(map(lambda x: x[1:],img_wise_data[:-1]))
        
        ##Converting the annotations which are in string to a list formaT. Eg shown below---
        '''
        [[[1440, 1057, 1540, 1158, 3],
        [1303, 868, 1403, 971, 3],
        [900, 578, 1008, 689, 3],
        [83, 73, 199, 186, 3]],
        [[14, 157, 150, 118, 3],
        [133, 868, 143, 71, 3],
        [900, 578, 108, 89, 3],
        [611, 304, 73, 48, 3],
        [881, 198, 3, 312, 3],
        [1480, 193, 574, 293, 3]]]
        '''
        y_true=[list(map(lambda x:list(map(int,x.split(','))) ,annot_for_one_image)) for annot_for_one_image in y_true]
        #max_boxes will tell us how many max boxes does the train data have ##TO be changed to make it constant
        max_boxes= max([len(i) for i in y_true])

        y_true= [i if len(i) == max_boxes else i+ [[0]*5]*(max_boxes-len(i)) for i in y_true]
        y_true= tf.convert_to_tensor(y_true, tf.float32)
        #img_tensors= tf.convert_to_tensor(list(map(read_image, img_path_list)))
        return tf.data.Dataset.from_tensor_slices((tf.constant(img_path_list), y_true))


parse_file('training.txt')

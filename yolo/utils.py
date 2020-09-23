import tensorflow as tf
import os


def read_and_resize_image(image_path,size= 416):
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


def transform_annotations(y_data,grid_size,anchors,mask):
    
    # y_out: (grid, grid, n_anchors, [x, y, w, h, obj, class])
    y_out = tf.zeros((grid_size, grid_size, 3, 6))
    
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    
    id_ =0
    for box in y_data:
        box_wh= box[...,2:4]- box[...,:2]
        min_wh= tf.minimum(anchors,box_wh)
        intersection = min_wh[...,0] *min_wh[...,1]
        anchors_area = anchors[...,0] * anchors[...,1]
        box_area= box_wh[...,0] * box_wh[...,1]
        iou= intersection/(anchors_area + box_area - intersection)
        index_for_anchor= tf.argmax(iou)
        
        if index_for_anchor in mask:
            box_xy= (box[...,2:4] + box[...,:2])/2
            
            modified_anchor= tf.cast(tf.math.floormod(index_for_anchor, 3), tf.int32)
            grid_cell_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
            
            index = tf.stack(
                    [grid_cell_xy[1], grid_cell_xy[0], modified_anchor])
            
            update = tf.concat((box_xy[0],box_xy[1], box_wh[0], box_wh[1],tf.constant(1.0),box[-1]),axis=0)
            
            indexes = indexes.write(id_, index)
            updates = updates.write(id_, update)
            #tf.print(indexes.stack())
            #tf.print(updates.stack())
            id_ += 1
            
    y_out = tf.tensor_scatter_nd_update(y_out, indexes.stack(), updates.stack())
    return y_out
                



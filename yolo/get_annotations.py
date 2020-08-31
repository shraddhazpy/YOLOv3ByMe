import json
import os

DATA_SUBDIR ='data/malaria'
JSONS = ['training.json','test.json']

CLASSES= ['difficult','gametocyte','leukocyte','red blood cell','ring','schizont','trophozoite']


path_main_dir = os.path.abspath(__file__+'/../..')
for json_file in JSONS:
    path_json= os.path.join(path_main_dir,DATA_SUBDIR,json_file)
    if os.path.isfile(path_json):
        with open(path_json,'r') as fp:
            train_or_test_json= json.loads(fp.read())

        
        with open(f"{json_file.rstrip('.json')}.txt",'w') as f:
            for image_data in train_or_test_json:
                image_path= os.path.join(os.path.dirname(path_json),image_data['image']['pathname'].lstrip('/'))
                f.write(image_path + '\n')

                for cell_data in list(map(lambda x: [x['bounding_box']['minimum']['c'],\
                x['bounding_box']['minimum']['r'],x['bounding_box']['maximum']['c'],\
                x['bounding_box']['maximum']['r'],CLASSES.index(x['category'])],\
                image_data['objects'])):
                    f.write(','.join(map(str,cell_data)) + '\n')

                f.write('\n')



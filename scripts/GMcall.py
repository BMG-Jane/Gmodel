#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Call the functions in class GeneralModel
#Call GMclass
#no arg, direct variables
#Jane Z., May 24, 2019

from GMclass import GeneralModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #parameters 
    org_model='VGG16' #could be: 'VGG16', 'Resnet50', 'DenseNet121', 'DenseNet169', 'DenseNet201'
    model_name_init = 'BMG5' #the name init of generated model
    lr = 0.001 #learning rate
    nb_epoch = 6 #number of epoches
    batch_size = 100 
    val_split = 0.2 # val_split to validation set, 1-val_split to train set
    
    optimizer = 'adam' #'adam' or 'sgd'
    
    
    
    ##init
    gmodel = GeneralModel()
    
    ##train
    train_dir = '/home/jane/train_egret' #train data folder
    csv_dir_train = '/home/jane/data_details_short.csv'#csv of train data
       
    #load csv file of train data
    data = gmodel.Load_csv(csv_dir_train)
    #Load train data，split to strain and validation sets
    train_generator, validation_generator = gmodel.Load_data_Trainset(data, train_dir, val_split, batch_size)
    #Data statistics of train and validation sets
    train_counter, val_counter = gmodel.Data_Stat(train_generator, validation_generator)
    #Model training
    model_path,model_name,model=gmodel.Model_Train(train_generator,validation_generator,org_model,model_name_init,lr,nb_epoch,optimizer)
    #Model saving
    gmodel.Model_Saving(model_path, model_name, model)
    
    ##evaluation
#    model_path = './BMG5'
#    model_name = 'BMG5_VGG16_100_6_0_224_adam_0.001'
    test_dir = '/home/jane/test_egret/data'
    csv_dir_test = '/home/jane/test_egret/data_details_test.csv'
    
    #load the trained model
    model = gmodel.Model_Loading(model_path, model_name)
    #load the csv file for evaluation
    data_test = gmodel.Load_csv(csv_dir_test)
    #load the data for evaluation
    test_generator = gmodel.Load_data_Testset(data_test,test_dir)
    #evaluation results
    eval_results, acc, f1, precision, recall, conf_matrix, auc, fpr,tpr = gmodel.Model_Evaluation(model, test_generator)
    plt.plot(fpr,tpr,marker='o')
    plt.show()
    
    ##inferance a sequence of images
    pred_dir = '/home/jane/test_egret' #images in the subfolder of pred_dir
    #load the sequence of images for prediction
    pred_generator = gmodel.Load_data_Predset(pred_dir)
    #prediction
    pred_results = gmodel.Model_Prediction(model, pred_generator)
    
    ##inferance one single image
    img_path = '/home/jane/test_egret/data/P_01700_RIGHT_CC.png'
    pred_one, pred_class = gmodel.Model_Prediction_img(model, img_path)
    


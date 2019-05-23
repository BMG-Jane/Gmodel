#!/usr/bin/env python
# coding: utf-8

# In[107]:

#test github
#training done
#evaluation done
#add prediction
#better organization 
#better flow
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras import applications
from keras.applications.densenet import DenseNet121, DenseNet169
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras import backend as K

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from PIL import ImageFile

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
from collections import Counter
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, classification_report


# In[108]:


class GeneralModel(object):
    def __init__(self):   
 #       self.train_dir=args.train_dir
 #       self.val_dir = args.val_dir
#        self.dst_dir = dst_dir
        
        self.model = 'VGG16' #args.model
        self.model_name = 'GM' #args.model_name
        self.lr = 0.001 #args.lr
        self.nb_epoch = 2 #args.nb_epoch
        self.batch_size = 100 #args.batch_size
        self.val_split = 0.3 #args.val_split
        self.optimizer = 'adam' #args.optimizer
        self.trainable_layers_final = 0 #args.trainable_layers
        self.fc_layer_size = 1024 #args.fc_layer_size
        self.steps_per_epoch = 100 #args.steps_per_epoch
        self.dst_dir='./'
        
        self.img_rows=224 #resolution of input for this model
        self.img_cols=224
        
      #  self.nb_classes=0
        self.nb_samples=0        
       

    def Data_Stat(self, train_generator, val_generator):
        '''data statistics'''
        print ('Number of classes: ', len(train_generator.class_indices))
        print ('Number of samples in train set: ', train_generator.n)
        print ('Number of samples in validation set: ', val_generator.n)
        list_classes_train = train_generator.class_indices
        list_classes_train = dict((v,k) for k,v in list_classes_train.items())
        class_namelist_train = [list_classes_train[k] for k in train_generator.labels]
        print('In train set, number of samples in each class:')        
        train_counter = dict(Counter(class_namelist_train))
        print (train_counter)
        list_classes_val = val_generator.class_indices
        list_classes_val = dict((s,u) for u,s in list_classes_val.items())
        class_namelist_val = [list_classes_val[u] for u in val_generator.labels]
        print('In test set, number of samples in each class:')
        val_counter = dict(Counter(class_namelist_val))
        print (val_counter)
        return train_counter, val_counter
      

    def Load_csv(self,csv_path):
        '''
           Load csv file 
        '''
        dataall = pd.read_csv(csv_path,header=None,names=['image','classid'])
        dataall2=dataall.dropna()
        data=dataall2.drop_duplicates('image','first')
        self.nb_classes=data.classid.nunique()
        print('loaded')
        return data
        
    #https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
    def Load_data_Trainset(self,data,train_path, val_split, batch_size):
        datagen = ImageDataGenerator(
            rescale = 1. / 255,            
            validation_split = val_split
        )
        
        self.batch_size = batch_size
        train_generator = datagen.flow_from_dataframe(
            dataframe=data,
            directory=train_path, #'/home/jane/test_data',
            x_col='image',
            y_col='classid',
            target_size=(self.img_rows, self.img_cols),
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=13,
            subset='training'
        )
        
        validation_generator = datagen.flow_from_dataframe(
            dataframe=data,
            directory= train_path, #'./',
            x_col='image',
            y_col='classid',
            target_size=(self.img_rows, self.img_cols),
            class_mode='categorical',
            batch_size = self.batch_size,
            shuffle=True,
            seed=13,
            subset='validation'        
        )        
                
        return train_generator, validation_generator
    
    def Load_data_Testset(self, data, test_path):
        test_datagen = ImageDataGenerator(rescale = 1. /255)
        print(test_path)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=data,
            directory = test_path,
            x_col='image',
            y_col='classid',
            target_size = (self.img_rows, self.img_cols),
            class_mode = 'categorical',#None,
            batch_size=1, #self.batch_size,
            shuffle=False,
            seed=13            
        )
        
        return test_generator
    
    def Load_data_Predset(self, pred_path):
        pred_datagen = ImageDataGenerator(rescale = 1. /255)
        
        pred_generator = pred_datagen.flow_from_directory(
            pred_path,
           # y_col=None,
            target_size = (self.img_rows, self.img_cols),
            class_mode = None,
            batch_size = 1,
            shuffle = False            
        )
        
        return pred_generator

    
    def Model_Train(self, args, train_generator, validation_generator):
        '''To train the model'''
        self.model = args.model
        self.model_name = args.model_name
        self.lr = args.lr
        self.nb_epoch = args.nb_epoch
     #   self.batch_size = args.batch_size
        self.val_split = args.val_split
        self.optimizer = args.optimizer
        self.trainable_layers_final = args.trainable_layers
        self.fc_layer_size = args.fc_layer_size
        self.steps_per_epoch = args.steps_per_epoch
        
        # Create the base pre-trained model
        if self.model == 'VGG16':
            self.trainable_layers = 15
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            self.base_model = applications.VGG16(weights='imagenet', include_top=False)        
        elif self.model == 'Resnet50':
            self.trainable_layers = 142
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            self.base_model = applications.ResNet50(weights='imagenet', include_top=False)       
        elif self.model == 'DenseNet121':
            self.trainable_layers = 200
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            base_model = DenseNet121(weights='imagenet', include_top=False)
        elif self.model == 'DenseNet169':
            self.trainable_layers = 200
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            self.base_model = DenseNet169(weights='imagenet', include_top=False)
        elif self.model == 'DenseNet201':
            self.trainable_layers = 200
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            self.base_model = DenseNet201(weights='imagenet', include_top=False)        
        else:
            self.trainable_layers = 15
           # img_rows, img_cols = 224, 224  # Resolution of inputs
            self.base_model = applications.VGG16(weights='imagenet', include_top=False)
       
        model_name = self.model_name + '_' + self.model + '_' + str(self.batch_size) + '_' + str(self.nb_epoch) + '_' + str(self.trainable_layers_final)                  + '_' + str(self.img_cols) + '_' + self.optimizer + '_' + str(self.lr)
        model_path = os.path.join(self.dst_dir, self.model_name)
       
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print(model_path)    
       
        #build the output layer
        model = self.add_new_last_layer()
            
        # transfer learning       
        self.setup_to_transfer_learn(model)
        
        # fine-tuning        
        self.setup_to_finetune(model)
        
        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(model_path, model_name + '_{epoch:03d}-{val_loss:.2f}.hdf5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            period=1)

        # Helper: TensorBoard
        tensorboard = TensorBoard(log_dir=model_path)

        history_ft = model.fit_generator(
            generator=train_generator,
            epochs=self.nb_epoch,
            steps_per_epoch= (self.nb_samples/self.batch_size)+1,
            validation_steps = (validation_generator.samples/self.batch_size) +1,
            validation_data=validation_generator,
            callbacks = [tensorboard, checkpointer, EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5)],
            class_weight = 'auto',
            workers = 5)

        self.plot_training(history_ft, os.path.join(model_path, model_name))
        return model_path, model_name, model

 #       scores = model.evaluate_generator(validation_generator, workers=5, steps = validation_generator.samples/self.batch_size +1)
 #       print('\n\nFinal Model Loss: ' + str(scores[0]) + '\nAcc: ' + str(scores[1]))

    def Model_Saving(self, model_path, model_name, model):
        '''save the model'''
        model.save(os.path.join(model_path, model_name + '.h5'))        
       

    def add_new_last_layer(self):
        """Add last layer to the convnet
        Args:
        base_model: keras model excluding top
        nb_classes: # of classes
        Returns:
        new keras model with last layer
        """
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.fc_layer_size, activation='relu', name='fc-1')(x)
        x = Dropout(0.3)(x)
        x = Dense(int(self.fc_layer_size / 2), activation='relu', name='fc-2')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.nb_classes, activation='softmax', name='output_layer')(x)
        model = Model(inputs=self.base_model.input, outputs=predictions)
        return model

    #def setup_to_transfer_learn(model, base_model):
    def setup_to_transfer_learn(self, model):
        """Freeze all layers and compile the model"""
        for layer in self.base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def setup_to_finetune(self, model):
        """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
        Args:
        model: keras model
        """
        # We chose to train the top 2 inception blocks, i.e. we will freeze the first 249 layers and unfreeze the rest.
        for layer in model.layers[:self.trainable_layers_final]:
            layer.trainable = False
        for layer in model.layers[self.trainable_layers_final:]:
            layer.trainable = True
        if self.optimizer == 'sgd':
            opt = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def plot_training(self,history, Model_Name):
        '''To plot the curves of the training procedure'''
        # summarize history for Accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Model_Name + '_Acc.png')
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(Model_Name + '_Loss.png')
        plt.close()
        
        
    def Model_Evaluation(self, model, test_generator):
        ''' To evaluate the performance of the model'''
        STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
       # model.evaluate_generator(generator=test_generator, steps = STEP_SIZE_TEST)
        test_generator.reset()
        Eval_result = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose=1)
        Eval_class_indices = np.argmax(Eval_result, axis = 1)
        print('evaluated class indices: ', Eval_class_indices)
        labels = (test_generator.class_indices)
        print('labels: ', labels)
        labels = dict((v,k) for k, v in labels.items())
        print('labels: ', labels)
        Evals = [labels[k] for k in Eval_class_indices]
        print('changeed k for evalutaed:', Evals)
        
        Gtruth = test_generator.classes
        print(Gtruth)
        filenames = test_generator.filenames
        results=pd.DataFrame({'Filename':filenames, 'Predictions':Evals,})
        print(results)
        
      #  print "\n\n\n", weights_path
        print ('Accuracy:', accuracy_score(Gtruth, Eval_class_indices))
        print ('F1 score:', f1_score(Gtruth, Eval_class_indices, average="macro"))
        print ('Precision:', precision_score(Gtruth, Eval_class_indices, average= "macro"))
        print ('Recall:', recall_score(Gtruth, Eval_class_indices, average = "macro"))
      #  print '\n clasification report:\n', classification_report(Gtruth, Eval_class_indices, target_names=class_labels)
        print ('\n confussion matrix:\n',confusion_matrix(Gtruth, Eval_class_indices))
        
    def Model_Prediction(self, model, pred_generator):
        '''To predict new images'''
        STEP_SIZE_PRED = pred_generator.n//pred_generator.batch_size
        pred_generator.reset()
        Pred_result = model.predict_generator(pred_generator, steps = STEP_SIZE_PRED, verbose = 1)
        print(Pred_result)
        Pred_class_indices = np.argmax(Pred_result, axis = 1)
        print(Pred_class_indices)
       
        
        
    def Model_Prediction_img(self,model,img_path):
        '''To predict one single image'''
        img=image.load_img(img_path,target_size=(self.img_rows,self.img_cols))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred_one = model.predict(x)     
        pred_class = np.argmax(pred_one)+1
        print('pred_one:',pred_one,'pred_class:',pred_class)
        return pred_one, pred_class

    def Model_Loading(self,model_path,model_name):
        '''To load the trianed model'''
        model = load_model(os.path.join(model_path, model_name) + '.h5')        
        return model
        




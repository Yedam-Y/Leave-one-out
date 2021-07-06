# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:20:14 2021

@author: CML_Ye
"""

def Model_Train(file_directory, sc, file_ID, epochs, batch_size, k_fold, val_n):

    tr_list = os.listdir(file_directory)
       
    for fold in range(k_fold):
        rand_list = sample(range(0, len(tr_list)), val_n)  # filelist : drug list
        Train_list = []
        Valid_list = []
        for i in range(len(tr_list)):
            # if i == rand_list[0] or i == rand_list[1]:
            if i == rand_list[0] :
                Valid_list.append(tr_list[i])
                # Valid_level.append(training_risk_level[i])  
            else:
                Train_list.append(tr_list[i])
                # Train_level.append(training_risk_level[i])
                
        tr, train_y = data_load2(file_directory , Train_list )
        tr1 = np.delete(tr, [1,5,6,10], axis =1)
        train = sc.transform(tr1)
        
        val, val_y = data_load2(file_directory , Valid_list )
        val1 = np.delete(val, [1,5,6,10], axis =1)
        valid = sc.transform(val1)       
        
        
        Model_path = 'C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/Model/'
        
        if not os.path.exists(Model_path):
            os.mkdir(Model_path)
            
        save_path = Model_path+str(fold)+'-{epoch:02d}-{loss:.4f}-{accuracy:.4f}--{val_loss:.4f}-{val_accuracy:.4f}-Model.hdf5'
        checkpoint = ModelCheckpoint(filepath = save_path, monitor = 'val_accuracy', save_best_only = True)
        callback_list = [checkpoint]
        
        Model = Drug_Model()
        Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
        hist = Model.fit(train, train_y, epochs = epochs,shuffle =True, batch_size = batch_size, validation_data = (valid, val_y), callbacks = callback_list, verbose = 1)
        
        print('################################'+ str(fold) + '#######################################')
        fig, loss_ax = py.subplots(figsize = (10,7))
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'r', linestyle ='--', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'b', linestyle ='--', label='val loss')

        acc_ax.plot(hist.history['accuracy'], 'r', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'b', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        
        py.savefig(Model_path+str(fold)+'.png', dpi=300)
        py.show()
        
    return Model


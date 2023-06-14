def detectLstm(data):
    X = data[[ 'TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']]
    y = data['STATUS']
    segments,labels = segmantation(X,y,window_length=36 )
    save_path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Saved models'
    loaded_model = tf.keras.models.load_model(save_path+"lstm_all_P98_R84_F90_model.h5")

    with open(save_path + 'normalization_params_LSTM.json', 'r') as f:
        normalization_params = json.load(f)
    loaded_min = np.array(normalization_params['min'])
    loaded_max = np.array(normalization_params['max'])
    normalized_new_data = (data - loaded_min) / (loaded_max - loaded_min)
    
    
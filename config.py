
class Config:
    DATA_DIR = "my_dataset/rlvs"
    PROCESSED_DATA_DIR = "my_dataset/rlvs/processed_data"

    T_FRAMES = 16      
    IMAGE_SIZE = 224          
    
    FEATURE_DIM = 1280        
    LSTM_HIDDEN_SIZE = 128 
    LSTM_LAYERS = 2            
    
    BATCH_SIZE = 8         
    GRAD_ACCUM_STEPS = 4      
    LEARNING_RATE = 0.0001     
    WEIGHT_DECAY = 0.001     
    PATIENCE = 5           
    MIN_DELTA = 0.001     
    EPOCHS = 100
    DEVICE = "cuda"           

cfg = Config()
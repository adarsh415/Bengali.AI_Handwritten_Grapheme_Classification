export IMG_HEIGHT=137
export IMG_WIDTH=236
export EPOCHS=2
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=8
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL='resnet34'
export TRAINING_FOLD_CSV='../input/train_folds.csv'

export TRAINING_FOLDS='(0,1,2,3)'
export VALIDATION_FOLDS='(4,)'
C:\\ProgramData\\Anaconda3\\envs\\pytorch_new\\python -m train

export TRAINING_FOLDS='(0,1,2,4)'
export VALIDATION_FOLDS='(3,)'
C:\\ProgramData\\Anaconda3\\envs\\pytorch_new\\python -m train

export TRAINING_FOLDS='(0,1,4,3)'
export VALIDATION_FOLDS='(2,)'
C:\\ProgramData\\Anaconda3\\envs\\pytorch_new\\python -m train

export TRAINING_FOLDS='(0,4,2,3)'
export VALIDATION_FOLDS='(1,)'
C:\\ProgramData\\Anaconda3\\envs\\pytorch_new\\python -m train

export TRAINING_FOLDS='(4,1,2,3)'
export VALIDATION_FOLDS='(0,)'
C:\\ProgramData\\Anaconda3\\envs\\pytorch_new\\python -m train





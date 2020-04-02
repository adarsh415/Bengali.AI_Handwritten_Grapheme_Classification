import os
import ast
import torch
import model_dispatcher
from dataset import BengaliDatasetTrain
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm


DEVICE = 'cpu'
TRAINING_FOLD_CSV = os.environ.get('TRAINING_FOLD_CSV')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH = int(os.environ.get('IMG_WIDTH'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAINING_FOLDS = ast.literal_eval(os.environ.get('TRAINING_FOLDS'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))
BASE_MODEL = os.environ.get('BASE_MODEL')


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets

    l1 = torch.nn.CrossEntropyLoss()(o1,t1)
    l2 = torch.nn.CrossEntropyLoss()(o2,t2)
    l3 = torch.nn.CrossEntropyLoss()(o3,t3)

    return (l1+l2+l3)/3

def train(dataset, data_loader, model, optimizer):
    model.train()

    for batch, data in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        optimizer.zero_grad()
        outputs = model(image)
        target = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, target)

        loss.backward()
        optimizer.step()

def evaluate(dataset, data_loader, model, optimizer):
    model.eval()
    final_loss = 0
    counter = 0

    for batch, data in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        outputs = model(image)
        target = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, target)
        final_loss += loss
        counter +=1
    return final_loss/counter

def main():
    model = model_dispatcher.MODEL_DISPATCHER[BASE_MODEL](True)
    #model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(TRAINING_FOLDS,
                                        IMG_HEIGHT,
                                        IMG_WIDTH,
                                        MODEL_MEAN,
                                        MODEL_STD)

    train_loader = dataloader.DataLoader(
        dataset= train_dataset,
        batch_size= TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliDatasetTrain(VALIDATION_FOLDS,
                                        IMG_HEIGHT,
                                        IMG_WIDTH,
                                        MODEL_MEAN,
                                        MODEL_STD)

    valid_loader = dataloader.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3, verbose=True)

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model, optimizer)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'{BASE_MODEL}_{VALIDATION_FOLDS[0]}.bin')

if __name__ == '__main__':
    main()

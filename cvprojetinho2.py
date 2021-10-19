import splitfolders
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
import os, os.path
import pandas as pd
import string    
import random 
import shutil
import torchvision
import time
from torch.optim import lr_scheduler

#Fazer tarefa 3 cm opencv e numpy, ver site do kaggle
#Pre processamento inicial
#Tarefa 2
#Renomeando tod
DIR1="todas_imagens\\Darth Vader"
files1=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
DIR2="todas_imagens\\Stormtrooper"
files2=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
DIR3="todas_imagens\\Yoda"
files3=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]

for i in files1:
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 30))    
    os.rename("todas_imagens\\Darth Vader\\"+i,str("todas_imagens\\Darth Vader\\"+ran+".jpg"))    

for i in files2:
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 30))    
    os.rename("todas_imagens\\Stormtrooper\\"+i,str("todas_imagens\\Stormtrooper\\"+ran+".jpg"))    

for i in files3:
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 30))    
    os.rename("todas_imagens\\Yoda\\"+i,str("todas_imagens\\Yoda\\"+ran+".jpg"))    

# define the random module  
S = 30  # number of characters in the string.  
# call random.choices() string module to find the string in Uppercase + numeric data.  
ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 30))    
print("The randomly generated string is : " + str(ran+".jpg")) # print the random data

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        #Lendo um pandas diretamente
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#Transforms
#Transformação das imagens
transform_imgs = transforms.Compose([
transforms.Grayscale(), transforms.CenterCrop(140)])


#Agora excluindo imagens problematicas
DIR1="todas_imagens\\Darth Vader"
darth=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
#np.array([0]*len(darth_train))
darth=pd.DataFrame(np.column_stack((darth,np.array([0]*len(darth)))),columns=["nome_arquivo","classe"])
darth_imgs=ImageDataset(darth , "todas_imagens\\Darth Vader", transform_imgs)
darth_imgs[0]
dataloader = torch.utils.data.DataLoader(darth_imgs, batch_size=len(darth_imgs), shuffle=False)
images, labels = next(iter(dataloader))

#Classe Stormtrooper
DIR1="todas_imagens\\Stormtrooper"
storm=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
storm=pd.DataFrame(np.column_stack((storm,np.array([0]*len(storm)))),columns=["nome_arquivo","classe"])
storm_imgs=ImageDataset(storm, DIR1, transform_imgs)
len(storm_imgs)
storm_imgs[36]

cont=0
for i in range(len(storm_imgs)):
    try:
        print(storm_imgs[i][0].shape) 
    except:
        cont=cont+1
        print(i)
        img_exc="todas_imagens\\Stormtrooper\\"+storm.iloc[i,0]
        os.remove(img_exc)
cont

storm=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
storm=pd.DataFrame(np.column_stack((storm,np.array([0]*len(storm)))),columns=["nome_arquivo","classe"])
storm_imgs=ImageDataset(storm, DIR1, transform_imgs)
len(storm_imgs)
dataloader = torch.utils.data.DataLoader(storm_imgs, batch_size=len(storm_imgs), shuffle=False)
images, labels = next(iter(dataloader))

#Classe Yoda
DIR1="todas_imagens\\Yoda"
yoda=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
yoda=pd.DataFrame(np.column_stack((yoda,np.array([0]*len(yoda)))),columns=["nome_arquivo","classe"])
yoda_imgs=ImageDataset(yoda, DIR1, transform_imgs)
len(yoda_imgs)
yoda_imgs[36]

cont=0
for i in range(len(yoda_imgs)):
    try:
        print(yoda_imgs[i][0].shape) 
    except:
        cont=cont+1
        print(i)
        img_exc="todas_imagens\\Yoda\\"+yoda.iloc[i,0]
        os.remove(img_exc)
cont

yoda=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
yoda=pd.DataFrame(np.column_stack((yoda,np.array([0]*len(yoda)))),columns=["nome_arquivo","classe"])
yoda_imgs=ImageDataset(yoda, DIR1, transform_imgs)
len(yoda_imgs)
dataloader = torch.utils.data.DataLoader(yoda_imgs, batch_size=len(yoda_imgs), shuffle=False)
images, labels = next(iter(dataloader))

images[0]
#FALTA MOSTRAR AS IMAGENS

#1 tarefa

splitfolders.ratio("todas_imagens", output="divisoes", seed=7, ratio=(0.7, 0.15,0.15))

#3 tarefa

#Criando dataframe com os rotulos

#Para rodar de uma vez
DIR1="divisoes\\train\\Darth Vader"
darth_train=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
darth_train=np.column_stack((darth_train,np.array([0]*len(darth_train))))

DIR2="divisoes\\train\\Stormtrooper"
storm_train=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
storm_train=np.column_stack((storm_train,np.array([1]*len(storm_train))))

DIR3="divisoes\\train\\Yoda"
yoda_train=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
yoda_train=np.column_stack((yoda_train,np.array([2]*len(yoda_train))))

train=pd.DataFrame(np.row_stack((darth_train, storm_train, yoda_train)),columns=["nome_arquivo","classe"])

#Dataset de validação
DIR1="divisoes\\val\\Darth Vader"
darth_val=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
darth_val=np.column_stack((darth_val,np.array([0]*len(darth_val))))

DIR2="divisoes\\val\\Stormtrooper"
storm_val=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
storm_val=np.column_stack((storm_val, np.array([1]*len(storm_val))))

DIR3="divisoes\\val\\Yoda"
yoda_val=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
yoda_val=np.column_stack((yoda_val, np.array([2]*len(yoda_val))))

val=pd.DataFrame(np.row_stack((darth_val, storm_val, yoda_val)),columns=["nome_arquivo","classe"])

#Dataset de teste
DIR1="divisoes\\test\\Darth Vader"
darth_test=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
darth_test=np.column_stack((darth_test,np.array([0]*len(darth_test))))

DIR2="divisoes\\test\\Stormtrooper"
storm_test=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
storm_test=np.column_stack((storm_test,np.array([1]*len(storm_test))))

DIR3="divisoes\\test\\Yoda"
yoda_test=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
yoda_test=np.column_stack((yoda_test,np.array([2]*len(yoda_test))))

test=pd.DataFrame(np.row_stack((darth_test, storm_test, yoda_test)),columns=["nome_arquivo","classe"])

#Dataset de treino
DIR1="divisoes\\train\\Darth Vader"
darth_train=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
for i in darth_train:  
    shutil.copyfile(DIR1+"\\"+i,"train\\"+i) 

darth_train=np.column_stack((darth_train,np.array([0]*len(darth_train))))

DIR2="divisoes\\train\\Stormtrooper"
storm_train=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
for i in storm_train:  
    shutil.copyfile(DIR2+"\\"+i,"train\\"+i) 

storm_train=np.column_stack((storm_train,np.array([1]*len(storm_train))))

DIR3="divisoes\\train\\Yoda"
yoda_train=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
for i in yoda_train:  
    shutil.copyfile(DIR3+"\\"+i,"train\\"+i)
    

#np.array([2]*len(yoda_train))
yoda_train=np.column_stack((yoda_train,np.array([2]*len(yoda_train))))

train=pd.DataFrame(np.row_stack((darth_train, storm_train, yoda_train)),columns=["nome_arquivo","classe"])


#Dataset de validação
DIR1="divisoes\\val\\Darth Vader"
darth_val=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
for i in darth_val:  
    shutil.copyfile(DIR1+"\\"+i,"val\\"+i)  

darth_val=np.column_stack((darth_val,np.array([0]*len(darth_val))))

DIR2="divisoes\\val\\Stormtrooper"
storm_val=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
for i in storm_val:  
    shutil.copyfile(DIR2+"\\"+i,"val\\"+i)
storm_val=np.column_stack((storm_val, np.array([1]*len(storm_val))))

DIR3="divisoes\\val\\Yoda"
yoda_val=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
for i in yoda_val:  
    shutil.copyfile(DIR3+"\\"+i,"val\\"+i)
yoda_val=np.column_stack((yoda_val, np.array([2]*len(yoda_val))))

val=pd.DataFrame(np.row_stack((darth_val, storm_val, yoda_val)),columns=["nome_arquivo","classe"])

#Dataset de teste
DIR1="divisoes\\test\\Darth Vader"
darth_test=[name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
for i in darth_test:  
    shutil.copyfile(DIR1+"\\"+i,"test\\"+i) 
darth_test=np.column_stack((darth_test,np.array([0]*len(darth_test))))

DIR2="divisoes\\test\\Stormtrooper"
storm_test=[name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
for i in storm_test:  
    shutil.copyfile(DIR2+"\\"+i,"test\\"+i) 

storm_test=np.column_stack((storm_test,np.array([1]*len(storm_test))))

DIR3="divisoes\\test\\Yoda"
yoda_test=[name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
for i in yoda_test:  
    shutil.copyfile(DIR3+"\\"+i,"test\\"+i)

yoda_test=np.column_stack((yoda_test,np.array([2]*len(yoda_test))))

test=pd.DataFrame(np.row_stack((darth_test, storm_test, yoda_test)),columns=["nome_arquivo","classe"])


#Falta colocar as imagens de treino, val e teste em pasta diferentes
#os.rename("a\\teste.txt","b\\teste.txt")    
#shutil.copyfile("a\\teste.txt","b\\teste.txt")
#Transforms
#transform_imgs = nn.Sequential(transforms.Grayscale(),
#                                transforms.CenterCrop(200))
#Testar pra ver se deu certo 
#Depois ver como colar automaticamenta as imagens em uma unica pasta    
train_imgs=ImageDataset(train , "train", transform_imgs)
dataloader_train = torch.utils.data.DataLoader(train_imgs, batch_size=len(train_imgs), shuffle=True)
train_imgs2, train_labels = next(iter(dataloader_train))

val_imgs=ImageDataset(val , "val", transform_imgs)
dataloader_val = torch.utils.data.DataLoader(val_imgs, batch_size=len(val_imgs), shuffle=True)
val_imgs2, val_labels = next(iter(dataloader_val))

test_imgs=ImageDataset(test , "test", transform_imgs)
dataloader_test = torch.utils.data.DataLoader(test_imgs, batch_size=len(test_imgs), shuffle=True)
test_imgs2, test_labels = next(iter(dataloader_test))

len(train_imgs)
len(val_imgs)
len(test_imgs)

X_train=np.array([train_imgs2[i].numpy().flatten() for i in range(len(train_imgs))])
X_val=np.array([val_imgs2[i].numpy().flatten() for i in range(len(val_imgs))])
X_test=np.array([test_imgs2[i].numpy().flatten() for i in range(len(test_imgs))])

images[1]
labels
images[0]
#Transformações das imagens
train_imgs[922]

train_imgs[0][0].numpy().flatten().shape
train_imgs[2][0].shape==torch.Size([1, 200, 200])
# Tarefa 3
dataset_train=train_imgs[0][0].numpy().flatten()

#X_cv=np.row_stack((X_train, X_val))
#y_cv=np.concatenate((train_labels, val_labels))
from sklearn.decomposition import PCA
pca=PCA(n_components=100)
pca.fit(X_train)
X_train2=pca.transform(X_train)
X_val2=pca.transform(X_val)
X_test2=pca.transform(X_test)
np.sum(pca.explained_variance_ratio_)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import RandomizedSearchCV

model=LogisticRegression(solver="lbfgs", max_iter=200, multi_class="multinomial")
model.fit(X_train2, train_labels)

#Calculando metricas
accuracy_score(train_labels, model.predict(X_train2))
recall_score(train_labels, model.predict(X_train2), average=None)
precision_score(train_labels, model.predict(X_train2), average=None)


accuracy_score(val_labels, model.predict(X_val2))
recall_score(val_labels, model.predict(X_val2), average=None)
precision_score(val_labels, model.predict(X_val2), average=None)

accuracy_score(test_labels, model.predict(X_test2))
recall_score(test_labels, model.predict(X_test2), average=None)
precision_score(test_labels, model.predict(X_test2), average=None)

# =============================================================================
# #4 Tarefa
# =============================================================================
#Ajeitar ordem da rede
#140-9=131
#131-9=122/2=61
#61-9=52/2=26
#Tentar melhorar
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 10)  
        self.conv2 = nn.Conv2d(5, 8, 10)  
        self.pool1 = nn.MaxPool2d(2, 2)   
        self.conv3 = nn.Conv2d(8, 2, 10) 
        self.pool2 = nn.MaxPool2d(4, 4)   
        self.fc1 = nn.Linear(2* 26 * 26, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), 1)
        return(x)


cnn = CNN()

#Treinament
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)


epochs = 2
for epoch in range(epochs):  
    #running_loss_train = 0.0
    #running_loss_val = 0.0
    #for i in range(len(labels)):
        # get the inputs; data is a list of [inputs, labels]

        # zera os gradientes de todas as variaveis
    optimizer.zero_grad()

    # forward + backward + optimize
    output_train = cnn(train_imgs2.float())
    output_val = cnn(val_imgs2.float())

    loss_train = criterion(output_train, torch.tensor(np.array(train_labels).astype("int64"))) # calcula loss da batch
    loss_val = criterion(output_val, torch.tensor(np.array(val_labels).astype("int64")))  
    
    loss_train.backward() # Computa o gradiente da loss com relação aos parametros do modelo
    optimizer.step() # Update dos parametros
    print(epoch + 1, "Loss de treino",np.round(loss_train.item(), 5)
          ,"|| Loss de val", np.round(loss_val.item(), 5))
    # print statistics
    #running_loss_train += loss_train.item()
    #running_loss_train += loss_val.item() 
    
    #loss_train
    # calcula loss da batch
    # Atualiza a validation loss média
    
    # Cálculo dos valores de loss médios
    #train_loss = running_loss_train
    #validation_loss = running_loss_val
    
    #if i+1 % 10 == 0:    # para cada  mini-batches de 2000
 
        #running_loss = 0.0
        
#Metricas no conjunto de teste
output = cnn(test_imgs2.float())
loss = criterion(output, torch.tensor(np.array(test_labels).astype("int64")))


pred = torch.max(output, 1)[1].numpy()
test_labels
#Recall ou precision da classe 0
pos1=np.where(pred==0)[0]

np.sum(np.array(test_labels).astype("int64")[pos1]==pred[pos1])/len(pos1)

#Classe 2
pos2=np.where(pred==1)[0]

np.sum(np.array(test_labels).astype("int64")[pos2]==pred[pos2])/len(pos2)

#Classe 3
pos3=np.where(pred==2)[0]

np.sum(np.array(test_labels).astype("int64")[pos3]==pred[pos3])/len(pos3)

#Tarefa 5
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in [next(iter(dataloader_train)), next(iter(dataloader_val))]:
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(labels)
            epoch_acc = running_corrects.double() / len(labels)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


#train
#Deu erro, nao consegui
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)

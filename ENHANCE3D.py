import torch
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torchmetrics
from torch import nn
import PIL
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass

class RandomCrop:
    def __init__(self, h, w):
        self.h, self.w = h, w

    def __call__(self, x):
        h, w, _ = x.shape
        h = h - self.hda
        w = w - self.w
        H = np.random.randint(0, h, 1)[0]
        W = np.random.randint(0, w, 1)[0]
        x = x[H: H + self.h, W: W + self.w, :]
        assert x.shape == (self.h, self.w, 3), "Size mismatch"
        return x

class Resize:
    def __init__(self, h, w, interpolation):
        self.h, self.w = h, w
        self.interpolation = interpolation

    def __call__(self, x):
        return cv2.resize(x, (self.h, self.w), interpolation=self.interpolation)

class Rescale:
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, x):
        return x / self.max_value

class TransformPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, x):
        for job in self.pipeline:
            x = job(x)
        return x

class AdjustDimension:
    def __init__(self):
        pass

    def __call__(self, x):
        return np.transpose(x, (2, 0, 1))

class CustomDataset(Dataset):
    def __init__(self, data, device, xPipe=None):
        self.data = data
        self.xtp = xPipe
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.as_tensor(
            self.xtp(cv2.imread(x)), dtype=torch.float32, device=self.device
        ), torch.as_tensor(y, dtype=torch.int64, device=self.device)
bad = r'C:\Users\ferra\PycharmProjects\pycharmProjects\venv\archive (15)\defected'
good = r'C:\Users\ferra\PycharmProjects\pycharmProjects\venv\archive (15)\no_defected'

good_samples = []
for img in os.listdir(good):
    img_path = os.path.join(good, img)
    good_samples.append([img_path, 0])

bad_samples = []
for img in os.listdir(bad):
    img_path = os.path.join(bad, img)
    bad_samples.append([img_path, 1])

TEST_SIZE = 0.2

good_samples = shuffle(good_samples)
good_len = len(good_samples)
bad_samples = shuffle(bad_samples)
bad_len = len(bad_samples)

test_good, train_good = good_samples[:int(good_len * TEST_SIZE)], good_samples[int(good_len * TEST_SIZE):]
test_bad, train_bad = bad_samples[:int(bad_len * TEST_SIZE)], bad_samples[int(bad_len * TEST_SIZE):]

train_data = train_good + train_bad
test_data = test_good + test_bad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 224, 224
BATCH_SIZE = 2

train_tfx_pipeline = TransformPipeline([
    Resize(H, W, cv2.INTER_AREA), Rescale(255), AdjustDimension()
])

train_dataset = CustomDataset(train_data, device, train_tfx_pipeline)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomDataset(test_data, device, train_tfx_pipeline)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

ModeDict = {
            'min': lambda t2, t1, threshold=0: t1-t2 > threshold,
            'max': lambda t2, t1, threshold=0: t2-t1 > threshold
           }
class EarlyStopping:
    def __init__(self,
                 monitor : str = 'val_loss',
                 patience : int = 0,
                 min_delta : float = 0.0,
                 mode : str = 'min',
                 restore : bool = True):
        assert patience >= 0
        assert mode in ModeDict.keys()

        self._monitor = monitor.lower()
        self._patience = patience
        self._restore = restore
        self._delta = min_delta
        self._mode = ModeDict[mode]
        self._baseline = None
        self.__counter = 0

    def __call__(self, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        history = kwargs['history']
        epoch = len(history['loss'])
        assert self._monitor in history.keys(), f'{self._monitor} not found'
        present = history[self._monitor][-1]
        satisfied, keep_training = True, True
        if self._baseline is None:
            self._baseline = present
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':present
            }, './checkpoint.pt')
        else:
            satisfied = self._mode(present, self._baseline, self._delta)
            if satisfied:
                print(f'{self._monitor} Updated: {present}')
                self._baseline = present
                self.__counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':present
                }, './checkpoint.pt')
            else:
                self.__counter += 1
                if self.__counter > self._patience:
                    keep_training = False
                    if self._restore:
                        checkpoint = torch.load('./checkpoint.pt')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return keep_training
def train_loop(dataloader, testloader, model, **conf):
    metrics = conf['metrics']
    criterion = conf['criterion']
    regularizers = conf['regularizers']
    callbacks = conf['callbacks']

    optimizer = conf['optimizer']
    max_iter = conf['max_iter']
    device = conf['device']

    history = dict()
    history['loss'] = []
    history['val_loss'] = []
    for m in metrics:
        history[m['name']] = []
        history[f'val_{m["name"]}'] = []

    for itr in range(max_iter):
        model.train()

        real_time = dict()
        real_time['loss'] = []
        for m in metrics:
            real_time[m['name']] = []
        for _, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)

            for regularizer in regularizers:
                loss += regularizer(model.parameters())

            real_time['loss'].append(loss.item())
            for m in metrics:
                real_time[m['name']].append(m['fn'](pred, y).item())


            loss.backward()
            optimizer.step()
        history['loss'].append(np.mean(real_time['loss']))
        epoch_loss = history['loss'][-1]
        print(f"\n[{itr:>5d}/{max_iter:>5d}]\tLoss: {epoch_loss:>4f}\t")

        for m in metrics:
            key = m['name']
            value = np.mean(real_time[m['name']])
            history[key].append(value)
            print(f"{key}: {value:>4f}", end=' ')

        model.eval()
        val_metrics = dict()
        val_metrics['val_loss'] = []
        for m in metrics:
            val_metrics[m['name']] = []
        with torch.no_grad():
            for X, y in testloader:
                pred = model(X)
                loss = criterion(pred, y)
                val_metrics['val_loss'].append(loss.item())
                for m in metrics:
                    val_metrics[m['name']].append(m['fn'](pred, y).item())
        print(f'\nValidation : Loss: {np.mean(val_metrics["val_loss"]):>4f}, ', end='')
        history['val_loss'].append(np.mean(val_metrics["val_loss"]))
        for m in metrics:
            key = f"val_{m['name']}"
            value = np.mean(val_metrics[m['name']])
            history[key].append(value)
            print(f"{key}: {value:>4f}", end=' ')
        break_training = False
        for callback in callbacks:
            if not callback(model=model, optimizer=optimizer, history=history):
                break_training = True
                break
        if break_training:
            print('Training end triggered by callback.')
            break
    return history

learning_rate = 1e-4
EPOCHS = 100
KERNELS = 64

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)
net.fc = nn.Linear(512, 2)

OPTIM = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)
callbacks = [EarlyStopping("val_loss", patience=10)]
regularizers = []

metrics = [
    {"name": 'Accuracy', "fn": torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)},
    {"name": 'Recall', "fn": torchmetrics.Recall(task='multiclass', num_classes=2).to(device)}
]

hist = train_loop(train_loader,
    test_loader,
    net,
    optimizer=OPTIM,
    max_iter=EPOCHS,
    metrics=metrics,
    criterion=criterion,
    regularizers=regularizers,
    callbacks=callbacks,
    device=device)


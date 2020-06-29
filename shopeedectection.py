
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.callbacks import SaveModelCallback
from torchvision.models import *
import pretrainedmodels
import os
from datetime import datetime

def fastai_gen_submission(path, model, suffix):
    testset = ImageList.from_folder('data/test/')
    learn = load_learner(path, file=model, test=testset)   
    learn.callback_fns = []
    preds, _ = learn.get_preds(DatasetType.Test)
    filenames = list(map(lambda x: os.path.basename(x), testset.items))
    pred_classes = preds.numpy().argmax(axis=1).tolist()
    preddf = pd.DataFrame({'filename': filenames, 'category':pred_classes})
    testdf = pd.read_csv('data/test.csv')
    submit = testdf.drop('category', axis=1).merge(preddf, on='filename')
    submit.to_csv(f'submit{suffix}.csv', index=False)


def fastai_learn(
        spl, 
        arch, 
        learn=None, 
        epochs=1, 
        imgsize=224, 
        lin_ftrs=[512], 
        bs=16, 
        lr=3e-3, 
        export=None, 
        cutoutholes=0, 
        xtransform=False):
    if arch is None and learn is None:
        print('Either arch or learn has to be given')
        return None

    data = ImageDataBunch.from_folder(
        path=f'{spl}/', 
        train='train', 
        valid='valid', 
        test='test', 
        ds_tfms=get_transforms(
            max_rotate=30.0,
            max_zoom=1.3,
            max_lighting = 0.3,
            max_warp = 0.3,
            xtra_tfms = [cutout(
                n_holes=(1,cutoutholes), 
                length=(int(0.1*imgsize), int(0.2*imgsize)))] 
            if cutoutholes > 0 else []
        ) if xtransform else get_transforms(), 
        size=imgsize, 
        bs=bs)
    data.normalize(imagenet_stats)
    if learn is None:
        learn = cnn_learner(data, arch, metrics=accuracy, lin_ftrs=lin_ftrs)
    else:
        learn.data=data

    learn.fit_one_cycle(epochs, max_lr=slice(lr))
    
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    pred_class = np.argmax(preds.numpy(),axis=1)
    labels = np.array(list(map(lambda x: int(x.as_posix().split('/')[-2]), learn.data.test_ds.items)))
    testaccuracy = sum(labels==pred_class)/len(labels)


    if export:
        learn.export(file=f'{export}.pkl')
    else:
        learn.export(file=f'res{datetime.now().strftime("%Y%m%d%H%M%S")}.pkl')

    with open(f'{spl}/accuracy.txt', 'a+') as f:
        f.write('{}: {}'.format(export if export is not None else '_', testaccuracy))

    return learn


def fastai_train_pipeline(
    arch, 
    lin_ftrs, 
    learn=None,
    transform=False, 
    cutoutholes=0, 
    name='test',
    epochs={10:5, 30:5, 100:5, 101: 5}):

    def train(spl, learn, sz, epochs, frozen=True):
        lr = 3e-3 if frozen else 6e-4
        return fastai_learn(spl, None, learn=learn, lr=lr, imgsize=sz, lin_ftrs=lin_ftrs, epochs=epochs)

    def train_tfms(spl, learn, sz, lastepochs, export=None, frozen=True):
        lr = 3e-3 if frozen else 6e-4
        return fastai_learn(spl, None, learn=learn, lr=lr, imgsize=sz, lin_ftrs=lin_ftrs, epochs=lastepochs, xtransform=True, cutoutholes=2, export=export)

    print('reload')
    
    if learn is None:
        spl = 'splsz256pc10'
        learn = fastai_learn(spl, arch, lin_ftrs=lin_ftrs, epochs=1)

    for sz in [256, 512, 768]:
    # for sz in [256]:
        for pct in [10, 30, 100]:
            ep = epochs[pct]
            if ep==0:
                continue
            spl = f'splsz{sz}pc{pct}'
            learn = train(spl, learn, sz*7//8, ep)
            learn.unfreeze()
            learn = train(spl, learn, sz*7//8, ep, frozen=False)
            learn.freeze()
        ep = epochs[101]
        if ep==0:
            continue
        learn = train_tfms(spl, learn, sz*7//8, ep)
        learn.unfreeze()
        learn = train_tfms(spl, learn, sz*7//8, ep, f'{name}-sz{sz}', frozen=False)
        learn.freeze()

        fastai_gen_submission('splsz256pc10', f'{name}-sz{sz}.pkl', f'-{name}-sz{sz}')

    return learn



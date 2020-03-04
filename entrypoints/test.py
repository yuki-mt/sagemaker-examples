from math import ceil
import argparse
import os
from typing import Tuple, Iterable, List, Dict, Any
import numpy as np
from glob import glob
import re
import json
import shutil

# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from keras import backend as K
from keras.models import Model
from keras.utils.data_utils import get_file


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, choices=['base'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)

    # SageMaker paremeters
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--data_dir', type=str, default='../dataset')

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    with open(os.path.join(os.path.dirname(__file__), 'config', args['config'] + '.json')) as f:
        default_args = json.load(f)
    return dict(default_args, **args)


def restore_checkpoint(pretrained_path: str, ckpt_path: str) -> Tuple[str, int, ModelCheckpoint]:
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_file_format = 'weights.{}.h5'
    ckpt_files = glob(os.path.join(ckpt_path, ckpt_file_format.format('*')))
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filepath=os.path.join(ckpt_path, ckpt_file_format.format('{epoch:03d}')),
        save_weights_only=True)

    if len(ckpt_files) == 0:
        filename = 'pretrained_weight.h5'
        s3_result = re.search('s3://(.+?)/(.+)', pretrained_path)
        if s3_result is not None:
            import boto3
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(s3_result.group(1))
            weights_path = os.path.join(ckpt_path, filename)
            bucket.download_file(s3_result.group(2), weights_path)
        elif pretrained_path.startswith('http'):
            weights_path = get_file('pretrained_weight.h5', pretrained_path, cache_subdir=ckpt_path)
        else:
            weights_path = os.path.join(ckpt_path, filename)
            shutil.copy(pretrained_path, weights_path)
        return weights_path, 0, checkpoint
    else:
        weights_path = ckpt_files[-1]
        search_result = re.compile(ckpt_file_format.format('([0-9]+)')).search(weights_path)
        if search_result is None:
            raise ValueError('saved weight files have invalid formats')
        epoch = int(search_result.group(1))
        return weights_path, epoch, checkpoint


def build_model(args: Dict[str, Any], weights_path: str) -> Model:
    K.clear_session()
    model = None  # build your own model
    # model.load_weights(weights_path, by_name=True)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(optimizer=adam, loss='', metrics=['accuracy'])
    return model


def train(args: Dict[str, Any], current_epoch: int, ckpt: ModelCheckpoint,
          model: Model, train_dataset_size: int, val_dataset_size: int,
          train_generator: Iterable[List[np.array]], val_generator: Iterable[List[np.array]]) -> History:
    def get_lr_schedule(lrs: Dict[str, float]):
        sorted_lrs = sorted([(int(e), lr) for e, lr in lrs.items()], key=lambda x: -x[0])

        def lr_schedule(epoch):
            for e, lr in sorted_lrs:
                if epoch >= e:
                    return lr
        return lr_schedule

    lr_scheduler = LearningRateScheduler(schedule=get_lr_schedule(args['lr']), verbose=1)
    callbacks = [ckpt, lr_scheduler]

    return model.fit_generator(generator=train_generator,
                               steps_per_epoch=ceil(train_dataset_size/args['batch_size']),
                               epochs=args['epochs'],
                               callbacks=callbacks,
                               validation_data=val_generator,
                               validation_steps=ceil(val_dataset_size/args['batch_size']),
                               initial_epoch=current_epoch)


def get_dataset(args: Dict[str, Any]) -> Tuple[Iterable[List[np.array]], Iterable[List[np.array]], int, int]:
    pass


def save(model: Model, output_path: str):
    # save the whole model
    output_model_path = os.path.join(output_path, "result")
    os.makedirs(output_model_path, exist_ok=True)
    model.save(os.path.join(output_model_path, 'models.h5'))

    # save the trained weight
    # output_weight_path = os.path.join(output_path, "weights")
    # os.makedirs(output_weight_path, exist_ok=True)
    # model.save_weights(os.path.join(output_weight_path, 'weights.h5'))


def main():
    args = get_args()
    weights_path, current_epoch, ckpt = restore_checkpoint(args['pretrained_path'], args['ckpt_path'])
    model = build_model(args, weights_path)
    train_generator, val_generator, train_dataset_size, val_dataset_size = get_dataset(args)
    history = train(args, current_epoch, ckpt, model, train_dataset_size,
                    val_dataset_size, train_generator, val_generator)
    save(model, args['output_path'])
    val_accuracy = history.history['val_acc'][-1]
    print('val_accuracy:', val_accuracy)

    # remove checkpoints
    for f in glob(os.path.join(args['ckpt_path'], '*')):
        os.remove(f)


if __name__ == '__main__':
    main()

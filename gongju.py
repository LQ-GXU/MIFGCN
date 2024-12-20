import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
from engine import trainer
import os
import pickle

import torch


def normalize_weather(tensor_list):
    normalized_tensors = []

    for tensor in tensor_list:
        # 获取最后一维的第四列和第五列
        column4 = tensor[:, :, :, 3]
        column5 = tensor[:, :, :, 4]

        # 计算均值和方差
        mean4 = column4.mean()
        std4 = column4.std()
        mean5 = column5.mean()
        std5 = column5.std()

        # 对第四列进行均值方差归一化
        column4_normalized = (column4 - mean4) / std4

        # 对第五列进行均值方差归一化
        column5_normalized = (column5 - mean5) / std5

        # 替换相应的列
        tensor[:, :, :, 3] = column4_normalized
        tensor[:, :, :, 4] = column5_normalized

        normalized_tensors.append(tensor)

    return normalized_tensors


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    units: int, week: 7 * 24, day: 24, recent(hour): 1

    points_per_hour: int, number of points per hour, depends on data

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, feature=1,merge=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 12, depends on data

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (17856, 170, 3)


    all_samples = []
    for idx in range(data_seq.shape[0]):  # idx=2016 begin to get sample   data_seq.shape[0]
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        hour_sample, target = sample  # (12, 170, 3), (12, 170, 3) (12*3, 170, 3)
        # all_samples.append((
        #     np.expand_dims(np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :], axis=2),  # (1, 170, 3, 12)
        #     np.expand_dims(np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :], axis=2),
        #     np.expand_dims(np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :], axis=2),
        #     np.expand_dims(target, axis=0).transpose((0, 2, 3, 1)))[:, :, 0, :]
        # ))

        all_samples.append((
            np.expand_dims(hour_sample, axis=0)[:, :, :, 0:feature],
            np.expand_dims(target, axis=0)[:, :, :, 0:1]
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]


    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_hour, train_target = training_set  # (9498, 170, 1, 12), same,(9498, 170, 1, 36) ,same  133,122,169,124
    val_hour, val_target = validation_set  # (3166, 170, 1, 12) 190,173,197,153
    test_hour, test_target = testing_set  # (3166, 170, 1, 12)

    # print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     train_week.shape, train_day.shape,
    #     train_hour.shape, train_target.shape))
    # print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    # print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    # (week_stats, train_week_norm,  # week_stats: {dict} mean:(1, 170, 3, 12)  std:(1, 170, 3, 12)
    #  val_week_norm, test_week_norm, train_week_mean, train_week_std) = normalization(train_week,
    #                                                 val_week,
    #                                                 test_week)
    #
    # (day_stats, train_day_norm,
    #  val_day_norm, test_day_norm, train_day_mean, train_day_std) = normalization(train_day,
    #                                               val_day,
    #                                               test_day)
    #
    # (recent_stats, train_recent_norm,  # recent_stats: {dict} mean:(1, 170, 3, 36)  std:(1, 170, 3, 36)
    #  val_recent_norm, test_recent_norm, train_recent_mean, train_recent_std) = normalization(train_hour,
    #                                                     val_hour,
    #                                                     test_hour)

    all_data = {
        'train': {

            'recent': train_hour,
            'target': train_target
        },
        'val': {

            'recent': val_hour,
            'target': val_target
        },
        'test': {

            'recent': test_hour,
            'target': test_target}}
    #     },
    #     'stats': {
    #         'week': week_stats,
    #         'day': day_stats,
    #         'recent': recent_stats,
    #         'week_mean': train_week_mean,
    #         'week_std': train_week_std,
    #         'day_mean': train_day_mean,
    #         'day_std': train_day_std,
    #         'recent_mean': train_recent_mean,
    #         'recent_std': train_recent_std
    #     }
    # }
    return all_data


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None

    # week_sample = np.concatenate([data_sequence[i: j]
    #                               for i, j in week_indices], axis=0)
    # day_sample = np.concatenate([data_sequence[i: j]
    #                              for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return hour_sample, target


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=(0, 1, 2), keepdims=True)  # （1，1,3,1）
    std = train.std(axis=(0, 1, 2), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm




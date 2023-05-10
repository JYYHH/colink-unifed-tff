# inner_epoch / loss_func

import sys
from time import sleep
import random
import subprocess
import json
import flbenchmark.datasets
import flbenchmark.logging
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import random
import pandas as pd
import time 
import flbenchmark.logging
import keras.backend as K
import nest_asyncio
from unifed.frameworks.tff.models import *
from unifed.frameworks.tff.evaluate import *
from unifed.frameworks.tff.data_loader import *
import tensorflow_addons as tfa
from random import sample
import os 

from tensorflow_federated.python.core.impl.executor_stacks.python_executor_stacks import sizing_executor_factory
from tensorflow_federated.python.core.impl.execution_contexts.sync_execution_context import ExecutionContext
from tensorflow_federated.python.core.impl.context_stack import set_default_context

def simulate_workload():
    argv = sys.argv
    # takes 3 args: mode(client/server), output and logging destination, Config
    if len(argv) != 6:
        raise ValueError(f'Invalid arguments. Got {argv}')
    role, participant_id, output_path, log_path, Config = argv[1:6]
    print('Simulated workload here begin.')
    simulate_logging(participant_id, role, Config)
    print(f"Writing to {output_path} and {log_path}...")
    with open(output_path, 'w') as f:
        f.write(f"Some output for {role} here.")
    with open(log_path, 'w') as f:
        with open(f"./log/{participant_id}.log", 'r') as f2:
            f.write(f2.read())
    # or, alternatively
    # with open(log_path, 'w') as f:
    #     f.write(f"Some log for {role} here.")
    print('Simulated workload here end.')


def simulate_logging(participant_id, role, Config):
    # source: https://github.com/AI-secure/FLBenchmark-toolkit/blob/166a7a42a6906af1190a15c2f9122ddaf808f39a/tutorials/logging/run_fl.py
    if role == 'server':
        config = json.loads(Config)

        # first download
        
        flbd = flbenchmark.datasets.FLBDatasets('~/flbenchmark.working/data')
        print("Downloading Data...")
        dataset_name = (
                        'student_horizontal',
                        'breast_horizontal',
                        'default_credit_horizontal',
                        'give_credit_horizontal',
                        'vehicle_scale_horizontal'
                        )
        for x in dataset_name:
            if config["dataset"] == x:
                train_dataset, test_dataset = flbd.fateDatasets(x)
                flbenchmark.datasets.convert_to_csv(train_dataset, out_dir='../csv_data/{}_train'.format(x))
                if x != 'vehicle_scale_horizontal':
                    flbenchmark.datasets.convert_to_csv(test_dataset, out_dir='../csv_data/{}_test'.format(x))
        vertical = (
            'breast_vertical',
            'give_credit_vertical',
            'default_credit_vertical'
        )
        for x in vertical:
            if config["dataset"] == x:
                my_dataset = flbd.fateDatasets(x)
                flbenchmark.datasets.convert_to_csv(my_dataset[0], out_dir='../csv_data/{}'.format(x))
                if my_dataset[1] != None:
                    flbenchmark.datasets.convert_to_csv(my_dataset[1], out_dir='../csv_data/{}'.format(x))
        leaf = (
            'femnist',
            'reddit',
            'celeba'
        )
        for x in leaf:
            if config["dataset"] == x:
                my_dataset = flbd.leafDatasets(x)

        # second run the training process
    
        nest_asyncio.apply()
        # config = json.load(open('config.json', 'r'))
        # print(__name__)
        dataset_name = ('breast_horizontal',
                        'default_credit_horizontal',
                        'give_credit_horizontal',
                        'student_horizontal',
                        'vehicle_scale_horizontal',
                        'femnist',
                        'reddit',
                        'celeba'
                        )

        feature_dimension = (30, 23, 10, 13, 18, 28 * 28, 10, 224 * 224 * 3)
        out_dimension = (2, 2, 2, 1, 4, 62, 10, 2)

        train_path = ['../csv_data/breast_horizontal_train/breast_homo_',
                    '../csv_data/default_credit_horizontal_train/default_credit_homo_',
                    '../csv_data/give_credit_horizontal_train/give_credit_homo_',
                    '../csv_data/student_horizontal_train/student_homo_',
                    '../csv_data/vehicle_scale_horizontal_train/vehicle_scale_homo_',
                    '~/flbenchmark.working/data/femnist/train/',
                    '~/flbenchmark.working/data/reddit/train/',
                    '~/flbenchmark.working/data/celeba/train/'
                        ]
        test_path = ['../csv_data/breast_horizontal_test/breast_homo_',
                    '../csv_data/default_credit_horizontal_test/default_credit_homo_',
                    '../csv_data/give_credit_horizontal_test/give_credit_homo_',
                    '../csv_data/student_horizontal_test/student_homo_',
                    '',
                    '~/flbenchmark.working/data/femnist/test/',
                    '~/flbenchmark.working/data/reddit/test',
                    '~/flbenchmark.working/data/celeba/test'
                        ]

        # config
        dataset_switch = dataset_name.index(config["dataset"])
        assert dataset_switch >= 0

        is_standard = False

        np.random.seed(0)
        random.seed(0)

        NUM_CLIENTS = config["training"]["client_per_round"]
        print('number of client per round = ',NUM_CLIENTS)
        NUM_EPOCHS = config["training"]["local_epochs"]
        # NUM_EPOCHS = 1
        BATCH_SIZE = config["training"]["batch_size"]

        print('Used dataset is ' + dataset_name[dataset_switch])

        if dataset_switch < 5:
            train_host = []
            if dataset_switch == 1:
                train_host = np.array(pd.read_csv(train_path[dataset_switch] + 'host_1.csv'))
            else:
                train_host = np.array(pd.read_csv(train_path[dataset_switch] + 'host.csv'))

            train_guest = np.array(pd.read_csv(train_path[dataset_switch] + 'guest.csv'))

            test_my = []
            if dataset_switch < 4:
                test_my = np.array(pd.read_csv(test_path[dataset_switch] + 'test.csv'))
            else:
                print('split the data manually into train and test in dataset:' + dataset_name[dataset_switch])
                test_my = np.vstack((train_host, train_guest))
                train_host, train_guest = train_host, train_guest

            host_dataset = tf.data.Dataset.from_tensor_slices(train_host[:,1:])
            guest_dataset = tf.data.Dataset.from_tensor_slices(train_guest[:,1:])
            test_dataset = tf.data.Dataset.from_tensor_slices(test_my[:,1:])

        elif dataset_switch == 5:
            train_client_data, test_client_data = read_data_femnist(train_path[dataset_switch], test_path[dataset_switch])
            train_client_data = [tf.data.Dataset.from_tensor_slices(x) for x in train_client_data]
            test_client_data = [tf.data.Dataset.from_tensor_slices(x) for x in test_client_data]

        elif dataset_switch == 6:
            train_client_data, test_client_data, reddit_outdim = read_data_reddit(train_path[dataset_switch], test_path[dataset_switch])
            train_client_data = [tf.data.Dataset.from_tensor_slices(x) for x in train_client_data]
            test_client_data = [tf.data.Dataset.from_tensor_slices(x) for x in test_client_data]   

        elif dataset_switch == 7:
            train_client_data, test_client_data = read_data_celeba(train_path[dataset_switch], test_path[dataset_switch])
            test_test = [tf.data.Dataset.from_tensor_slices(test_client_data[0])]
        else:
            assert 1 == 0

        print("Dataset initializing is already done...")


        # ----- preprocessing the data -----------------------------------------------

        print(' Epoch Number = {}, Batch Size = {}'.format(NUM_EPOCHS, BATCH_SIZE))
        print(' Notice that in this context, EPOCHS is equal to inner step....')

        # SHUFFLE_BUFFER = 100
        # PREFETCH_BUFFER = 10
        if dataset_switch < 5:
            def preprocess(dataset):
                def batch_format_fn(element):
                    return collections.OrderedDict(
                        x=tf.reshape(element[1:], [feature_dimension[dataset_switch]]),
                        y=tf.reshape(element[0], [1]))
                return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
            train_data = [preprocess(host_dataset), preprocess(guest_dataset)]
            test_data = [preprocess(test_dataset), preprocess(test_dataset)]
        elif dataset_switch == 5:
            if config['model'] == 'lenet':
                paddings = tf.constant([[2, 2,], [2, 2]])
                def preprocess(dataset):
                    def batch_format_fn(element):
                        return collections.OrderedDict(
                            x=tf.expand_dims(tf.pad(tf.reshape(element[1:], [28, 28]), paddings, "REFLECT"), -1),
                            y=tf.reshape(element[0], [1]))
                    return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
                train_data = [preprocess(x) for x in train_client_data]
                test_data = [preprocess(x) for x in test_client_data]
            else:
                def preprocess(dataset):
                    def batch_format_fn(element):
                        return collections.OrderedDict(
                            x=tf.reshape(element[1:], [feature_dimension[dataset_switch]]),
                            y=tf.reshape(element[0], [1]))
                    return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
                train_data = [preprocess(x) for x in train_client_data]
                test_data = [preprocess(x) for x in test_client_data]       
        elif dataset_switch == 6:
            def preprocess(dataset):
                def batch_format_fn(element):
                    return collections.OrderedDict(
                        x=element[0],
                        y=element[1] )
                return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
            train_data = [preprocess(x) for x in train_client_data]
            test_data = [preprocess(x) for x in test_client_data]   
        elif dataset_switch == 7:
            if config['model'][:3] != 'mlp':
                def preprocess(dataset):
                    def batch_format_fn(element):
                        return collections.OrderedDict(
                            x=tf.reshape(element[1:], [224, 224, 3]),
                            y=tf.reshape(element[0], [1]))
                    return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
                test_data = [preprocess(x) for x in test_test]
            else:
                def preprocess(dataset):
                    def batch_format_fn(element):
                        return collections.OrderedDict(
                            x=tf.reshape(element[1:], [feature_dimension[dataset_switch]]),
                            y=tf.reshape(element[0], [1]))
                    return dataset.map(batch_format_fn).repeat(NUM_EPOCHS).batch(BATCH_SIZE)
                test_data = [preprocess(x) for x in test_test]   


        print("data-preprocessing is done...")    


        def model_fn():
            # Select your model 
            if config["model"] == "logistic_regression":
                keras_model = lr(
                    in_dim = feature_dimension[dataset_switch], 
                    out_dim = out_dimension[dataset_switch],
                    is_standard = is_standard)
            elif config["model"][:3] == "mlp":
                keras_model = mlp(
                    in_dim = feature_dimension[dataset_switch], 
                    out_dim = out_dimension[dataset_switch],
                    hidden = [int(x) for x in list(config["model"].split('_'))[1:]])
            elif config["model"] == "linear_regression":
                keras_model = linear_regression(
                    in_dim = feature_dimension[dataset_switch])
            elif config["model"] == "lenet":
                keras_model = lenet(out_dimension[dataset_switch])
            elif config["model"] == "lstm":
                keras_model = lstm(reddit_outdim)
            elif config["model"] == "alexnet":
                keras_model = alexnet()
            else :
                assert 1 == 0

            config["training"]["loss_func"] = "cross_entropy"

            if config["training"]["loss_func"] == "cross_entropy":
                return tff.learning.from_keras_model(
                    keras_model,
                    input_spec=test_data[0].element_spec,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            elif config["training"]["loss_func"] == "mse" or config["training"]["loss_func"] == "MSE":
                return tff.learning.from_keras_model(
                    keras_model,
                    input_spec=test_data[0].element_spec,
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError()])


        # record the communication part
        def format_size(size):
            size = float(size)
            size /= 8.0
            return size

        def set_sizing_environment():
            sizing_factory = sizing_executor_factory()
            context = ExecutionContext(executor_fn=sizing_factory)
            set_default_context.set_default_context(context)
            return sizing_factory


        # Create your FedAvg Processing

        if config["algorithm"] == 'fedavg':
            print('Using FedAvg Algorithm.......')
            if config["training"]["optimizer"] == 'sgd':
                iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                    model_fn,
                    client_optimizer_fn=lambda: tfa.optimizers.SGDW(learning_rate=config["training"]["learning_rate"], weight_decay=config["training"]["optimizer_param"]["weight_decay"], momentum=config["training"]["optimizer_param"]["momentum"], nesterov=config["training"]["optimizer_param"]["nesterov"]),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
                    use_experimental_simulation_loop=True)
            elif config["training"]["optimizer"] == 'adam' or config["training"]["optimizer"] == 'Adam':
                iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                    model_fn,
                    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
                    use_experimental_simulation_loop=True)  
        elif config["algorithm"][:7] == 'fedprox':
            print('Using FedProx Algorithm.......')
            if config["training"]["optimizer"] == 'sgd':
                iterative_process = tff.learning.algorithms.build_weighted_fed_prox(
                    model_fn,
                    client_optimizer_fn=lambda: tfa.optimizers.SGDW(learning_rate=config["training"]["learning_rate"], weight_decay=config["training"]["optimizer_param"]["weight_decay"], momentum=config["training"]["optimizer_param"]["momentum"], nesterov=config["training"]["optimizer_param"]["nesterov"]),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
                    proximal_strength=float(config["algorithm"].split('_')[-1]),
                    use_experimental_simulation_loop=True)
            elif config["training"]["optimizer"] == 'adam' or config["training"]["optimizer"] == 'Adam':
                iterative_process = tff.learning.algorithms.build_weighted_fed_prox(
                    model_fn,
                    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
                    proximal_strength=float(config["algorithm"].split('_')[-1]),
                    use_experimental_simulation_loop=True) 
        elif config["algorithm"] == 'fedsgd':
            print('Using FedSGD Algorithm.......')
            if config["training"]["optimizer"] == 'sgd':
                iterative_process = tff.learning.algorithms.build_fed_sgd(
                    model_fn,
                    server_optimizer_fn=lambda: tfa.optimizers.SGDW(learning_rate=config["training"]["learning_rate"], weight_decay=config["training"]["optimizer_param"]["weight_decay"], momentum=config["training"]["optimizer_param"]["momentum"], nesterov=config["training"]["optimizer_param"]["nesterov"]), 
                    use_experimental_simulation_loop=True)
            elif config["training"]["optimizer"] == 'adam' or config["training"]["optimizer"] == 'Adam':
                iterative_process = tff.learning.algorithms.build_fed_sgd(
                    model_fn,
                    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
                    use_experimental_simulation_loop=True)  
        else:
            assert 1==0


        # ---------------------- Begin to Train -------------------------------------

        print("Initializing...........")
        environment = set_sizing_environment()
        state = iterative_process.initialize()
        NUM_ROUNDS = config["training"]["global_epochs"]
        print("TOTAL Training round is {}".format(NUM_ROUNDS))
        if config["model"] == "logistic_regression":
            my_keras_model = lr(
                in_dim = feature_dimension[dataset_switch], 
                out_dim = out_dimension[dataset_switch],
                is_standard = is_standard)
        elif config["model"][:3] == "mlp":
            my_keras_model = mlp(
                in_dim = feature_dimension[dataset_switch], 
                out_dim = out_dimension[dataset_switch],
                hidden = [int(x) for x in list(config["model"].split('_'))[1:]])
        elif config["model"] == "linear_regression":
            my_keras_model = linear_regression(
                in_dim = feature_dimension[dataset_switch])
        elif config["model"] == "lenet":
            my_keras_model = lenet(out_dimension[dataset_switch])
        elif config["model"] == "lstm":
            my_keras_model = lstm(reddit_outdim)
        elif config["model"] == "alexnet":
            my_keras_model = alexnet()
        else :
            assert 1 == 0

        print(my_keras_model.summary())

        final_time = 0
        pre_agg, pre_br = 0, 0

        with flbenchmark.logging.Logger(id=0, agent_type='aggregator') as logger:
            with logger.training():
                begin_time = time.time()

                for round_num in range(1, NUM_ROUNDS + 1):
                    print('Round : ',round_num)
                    with logger.training_round() as t:
                        t.report_metric('client_num', NUM_CLIENTS)

                        if dataset_switch < 7:
                            round_train_data = sample(train_data, NUM_CLIENTS)
                        else:
                            train_data = sample(train_client_data, NUM_CLIENTS)
                            train_data = [preprocess(tf.data.Dataset.from_tensor_slices(x)) for x in train_data]
                            round_train_data = train_data

                        # if config["algorithm"] == 'fedavg':
                        #     state, metrics = iterative_process.next(state, round_train_data)
                        if config["algorithm"][:7] == 'fedprox' or config["algorithm"] == 'fedsgd' or config["algorithm"] == 'fedavg':
                            x = iterative_process.next(state, round_train_data)
                            state, metrics = x.state, x.metrics
                        else:
                            assert 1==0

                        size_info = environment.get_size_info()
                        broadcasted_bits = size_info.broadcast_bits[-1]
                        aggregated_bits = size_info.aggregate_bits[-1]
                        true_agg, true_br = aggregated_bits - pre_agg, broadcasted_bits - pre_br
                        pre_agg, pre_br = aggregated_bits, broadcasted_bits

                        with logger.communication(target_id=-1) as c:
                            c.report_metric('byte', int(format_size(true_agg)))
                        with logger.communication(target_id=-1) as c:
                            c.report_metric('byte', int(format_size(true_br)))

                    if round_num == NUM_ROUNDS :
                        # if config["algorithm"] == 'fedavg':
                        #     state.model.assign_weights_to(my_keras_model)
                        if config["algorithm"][:7] == 'fedprox' or config["algorithm"] == 'fedsgd' or config["algorithm"] == 'fedavg':
                            iterative_process.get_model_weights(state).assign_weights_to(my_keras_model)
                        else:
                            assert 1==0

                    if dataset_switch == 7:
                        for x in round_train_data:
                            del x 
                            
                total_time = time.time() - begin_time
                print('Train uses time : ' + str(total_time) + ' s')

            with logger.model_evaluation() as e:
                if dataset_switch == 3:
                    x = auto_evaluate(my_keras_model, test_data, is_regression = (dataset_switch == 3), compute_auc = False)
                    e.report_metric('mse', float(x))
                elif dataset_switch <=3:
                    e.report_metric('auc', float(auto_evaluate(my_keras_model, test_data, is_regression = (dataset_switch == 3), compute_auc = True)))
                else:
                    if dataset_switch < 7:
                        e.report_metric('accuracy', float(auto_evaluate(my_keras_model, test_data, is_regression = (dataset_switch == 3), compute_auc = False, ignore_pad = (config["dataset"] == "reddit"))))
                    else:
                        metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
                        for x in test_client_data:
                            client_data = preprocess(tf.data.Dataset.from_tensor_slices(x))
                            for batch in client_data:
                                predictions = my_keras_model(batch['x'])
                                metric.update_state(y_true=batch['y'], y_pred=predictions)
                        e.report_metric('accuracy', float(metric.result()))

    elif role == 'client':
        pass # tff doesn't have the distributed version in present
    else:
        raise ValueError(f'Invalid role {role}')

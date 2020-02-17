import tensorflow as tf
import numpy as np
import os
import socket
import wandb
import multiprocessing
import argparse

from datagenerator import DataGenerator
from model import SpeechBaselineModel

TOTAL_SAMPLES = 2676
SEQ_LENGTH = 100
VECTOR_SIZE = 129


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--LSTM_1_SIZE", help="Hidden size for the first LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_2_SIZE", help="Hidden size for the second LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_3_SIZE", help="Hidden size for the third LSTM Layer", type=int, default=256)

    parser.add_argument(
        "--LSTM_4_SIZE", help="Hidden size for the fourth LSTM Layer", type=int, default=256)

    parser.add_argument('--learning_rate', help='Learning rate for optimizer',
                        type=float, default=0.01)

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=32)

    parser.add_argument('--worker_count', help='Number of workers for fit_generator',
                        type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--max_queue_size', help='Max queue size for fit_generator',
                        type=int, default=32 * 8)

    parser.add_argument('--use_multiprocessing', help='Use multiprocessing for fit_generator',
                        type=bool, default=False)

    args = parser.parse_args()

    return args


def main():
    wandb_tags = [socket.gethostname()]
    wandb.init(project="pytorch-lstm-audio", tags=','.join(wandb_tags))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.random.set_seed(0)
    np.random.seed(0)

    args = parse_args()

    model = SpeechBaselineModel(total_samples=TOTAL_SAMPLES)
    model.build(seq_length=SEQ_LENGTH, feature_dim=VECTOR_SIZE,
                lstm1_size=args.LSTM_1_SIZE, lstm2_size=args.LSTM_2_SIZE, lstm3_size=args.LSTM_3_SIZE, lstm4_size=args.LSTM_4_SIZE)
    model.compile(learning_rate=args.learning_rate)

    train_gen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=args.batch_size, train_set=True)
    val_gen = DataGenerator(
        'data/dev-noise-subtractive-250ms-1', seq_length=SEQ_LENGTH, batch_size=args.batch_size, test_set=True)

    model.train(train_gen=train_gen, val_gen=val_gen,
                batch_size=args.batch_size, epochs=args.epochs, worker_count=args.worker_count, max_queue_size=args.max_queue_size, use_multiprocessing=args.use_multiprocessing)


if __name__ == '__main__':
    main()

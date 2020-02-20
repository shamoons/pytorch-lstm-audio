import tensorflow as tf
import numpy as np
import os
import socket
import wandb
import multiprocessing
import argparse

from datagenerator import DataGenerator
from model import SpeechBaselineModel

VECTOR_SIZE = 161
NORMALIZER = {'mean': 1e-07, 'std': 1e-5}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for corrupted audio',
                        default='data/dev-noise-subtractive-250ms-1')

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

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        type=int, default=100)

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

    train_gen = DataGenerator(
        args.audio_path, seq_length=args.seq_length, batch_size=args.batch_size, train_set=True, normalizer=NORMALIZER)
    val_gen = DataGenerator(
        args.audio_path, seq_length=args.seq_length, batch_size=args.batch_size, test_set=True, normalizer=NORMALIZER)

    model = SpeechBaselineModel(total_samples=train_gen.count_files())
    model.build(seq_length=args.seq_length, feature_dim=VECTOR_SIZE,
                lstm1_size=args.LSTM_1_SIZE, lstm2_size=args.LSTM_2_SIZE, lstm3_size=args.LSTM_3_SIZE, lstm4_size=args.LSTM_4_SIZE)
    model.compile(learning_rate=args.learning_rate)

    model.train(train_gen=train_gen, val_gen=val_gen,
                batch_size=args.batch_size, epochs=args.epochs, worker_count=args.worker_count, max_queue_size=args.max_queue_size, use_multiprocessing=args.use_multiprocessing)


if __name__ == '__main__':
    main()

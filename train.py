import tensorflow as tf
import numpy as np
import os
import socket
import wandb
import multiprocessing
import argparse
import distutils
from datagenerator import DataGenerator
from model import SpeechBaselineModel

N_MELS = 161


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', help='Path for corrupted audio',
                        default='data/dev-noise-subtractive-250ms-1')

    parser.add_argument(
        "--LSTM1_SIZE", help="Hidden size for the first LSTM Layer", type=int, default=128)

    parser.add_argument(
        "--LSTM2_SIZE", help="Hidden size for the second LSTM Layer", type=int, default=128)

    parser.add_argument(
        "--LSTM3_SIZE", help="Hidden size for the third LSTM Layer", type=int, default=128)

    parser.add_argument(
        "--LSTM4_SIZE", help="Hidden size for the fourth LSTM Layer", type=int, default=128)

    parser.add_argument(
        "--LSTM5_SIZE", help="Hidden size for the fourth LSTM Layer", type=int, default=128)

    parser.add_argument('--learning_rate', help='Learning rate for optimizer',
                        type=float, default=0.01)

    parser.add_argument('--seq_length', help='Length of sequences of the spectrogram',
                        type=int, default=100)

    parser.add_argument('--epochs', help='Epochs to run',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='Batch size',
                        type=int, default=64)

    parser.add_argument('--worker_count', help='Number of workers for fit_generator',
                        type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--max_queue_size', help='Max queue size for fit_generator',
                        type=int, default=32 * 8)

    parser.add_argument('--use_multiprocessing', help='Use multiprocessing for fit_generator',
                        type=lambda x: bool(distutils.util.strtobool(x)), default=False)

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
        args.audio_path, seq_length=args.seq_length, batch_size=args.batch_size, train_set=True, n_mels=N_MELS)
    val_gen = DataGenerator(
        args.audio_path, seq_length=args.seq_length, batch_size=args.batch_size, test_set=True, n_mels=N_MELS)

    model = SpeechBaselineModel(total_samples=train_gen.count_files())
    model.build(seq_length=args.seq_length, feature_dim=N_MELS,
                lstm1_size=args.LSTM1_SIZE, lstm2_size=args.LSTM2_SIZE, lstm3_size=args.LSTM3_SIZE, lstm4_size=args.LSTM4_SIZE, lstm5_size=args.LSTM5_SIZE)
    model.compile(learning_rate=args.learning_rate)

    model.train(train_gen=train_gen, val_gen=val_gen,
                batch_size=args.batch_size, epochs=args.epochs, worker_count=args.worker_count, max_queue_size=args.max_queue_size, use_multiprocessing=args.use_multiprocessing)


if __name__ == '__main__':
    main()

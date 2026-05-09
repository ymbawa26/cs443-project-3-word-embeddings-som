'''skipgram.py
The Skipgram neural network
Yazan and Joshua
CS 443: Bio-Inspired Machine Learning
Project 3: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import time
import os
import numpy as np
import tensorflow as tf

import network
from layers import Dense
from skipgram_layers import Embedding
from tf_util import arange_index


class Skipgram(network.DeepNetwork):
    '''Skipgram neural network that learns word embeddings. It consists of the following structure:

    Input → Embedding (linear) → Dense (softmax)

    Both the input and output layer have `vocab_sz` units.

    The output layer uses regular softmax activation.

    The layers use He weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96):
        '''Skipgram constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: for text data, input_feats_shape=(vocab_sz,).
        C: int.
            Number of classes in the dataset.
        embedding_dim: int.
            The number of units in the Embedding hidden layer (H).

        Calls the superclass constructor and builds the Skipgram network.
        '''
        super().__init__(input_feats_shape)
        self.C = int(C)
        self.embedding_dim = int(embedding_dim)

        self.embedding = Embedding('Embed_0', self.embedding_dim)
        # Match the notebook's reference behavior without changing the reused base layer defaults globally.
        self.embedding.wt_scale = np.sqrt(1.0 / (2.0 * self.input_feats_shape[0]))
        self.output_layer = Dense('Dense_1', self.C, activation='softmax',
                                  prev_layer_or_block=self.embedding,
                                  wt_scale=np.sqrt(2.0 / self.embedding_dim))

    def __call__(self, x):
        '''Forward pass through the Skipgram with the data samples `x`.

        Parameters:
        -----------
        x: tf.int32 tensor. shape=(B,).
            Data sample/word INDICES.

        Returns:
        --------
        tf.float32 tensor. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        net_act = self.embedding(x)
        net_act = self.output_layer(net_act)
        return net_act

    def fit(self, x, y, batch_size=256, epochs=10, print_every=1, linear_lr_decay=True, linear_lr_min_lr=1e-5,
            verbose=True):
        '''Trains Skipgram on pairs of context word indices (samples `x`) and target word indices (labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.int32. shape=(N,).
            Data samples / context word indices in the vocab.
        y: tf.constant. tf.int32. shape=(N,).
            Labels / target word indices in the vocab.
        batch_size: int.
            Number of samples to include in each mini-batch.
        epochs: int.
            Network should train this many epochs.
        print_every: int.
            How often (in MINI-BATCHES) should the network print progress and record the training loss?
        linear_lr_decay: bool.
            Do we apply a linear learning rate decay on every MINI-BATCH?
        linear_lr_min_lr: float.
            The minimum allowable learning rate if doing lr decay. We do not allow the lr to go below this value.
        verbose: bool.
            If set to `False`, there should be no print outs during training. Messages indicating start and end of
            training are fine.

        Returns:
        -----------
        train_loss_hist: Python list of floats.
            Training loss averaged over the most recent `print_every` MINI-BATCHES.
            For example: if `print_every`=5000, train_loss_hist looks like:
            [avg_loss(batches 0-4999), avg_loss(batches 5000-9999), ...]

        Uses a simplified training loop with optional linear learning rate decay. `train_loss_hist`
        contains training losses averaged over every `print_every` chunk of mini-batches.
        For example, if `print_every` is 500, then losses obtained from mini-batches 0-499 would be averaged and added
        as ONE entry in `train_loss_hist`, then the next 500 mini-batch losses (500-999) would be averaged then added as
        ONE entry in `train_loss_hist`, and so on.
        '''
        N = len(x)
        mini_batches = int(np.ceil(N / batch_size))
        num_steps = epochs * mini_batches
        initial_lr = float(self.opt.learning_rate.numpy())

        # Define loss tracking containers
        train_loss_hist = []
        running_loss = 0.0
        running_count = 0
        rng = np.random.default_rng(seed=0)
        self.set_layer_training_mode(True)

        if verbose:
            print("Starting training...")

        step = 0
        e = 0
        for e in range(epochs):
            t0 = time.time()
            for _ in range(mini_batches):
                batch_inds = rng.integers(low=0, high=N, size=batch_size)
                batch_inds_tf = tf.convert_to_tensor(batch_inds, dtype=tf.int32)
                x_batch = tf.gather(x, batch_inds_tf)
                y_batch = tf.gather(y, batch_inds_tf)

                batch_loss = self.train_step(x_batch, y_batch)
                batch_loss = float(batch_loss.numpy())
                running_loss += batch_loss
                running_count += 1

                if linear_lr_decay:
                    self.lr_linear_decay(initial_lr, step, num_steps, linear_lr_min_lr)

                if (step + 1) % print_every == 0:
                    avg_loss = running_loss / running_count
                    train_loss_hist.append(avg_loss)
                    if verbose:
                        dt = time.time() - t0
                        print(f'Epoch {e + 1}/{epochs} | batch {step + 1}/{num_steps} | '
                              f'{dt:.2f}s | train_loss={avg_loss:.6f}')
                    running_loss = 0.0
                    running_count = 0

                step += 1

        if running_count > 0:
            train_loss_hist.append(running_loss / running_count)

        if verbose:
            print(f'Finished training after {e + 1} epochs!')
        return train_loss_hist

    def lr_linear_decay(self, initial_lr, t, num_steps, min_allowed_lr=1e-5):
        '''Applies a linear learning rate decay to the optimizer's learning rate on the MINI-BATCH level.

        See notebook for a refresher on the equation.

        Parameters:
        -----------
        initial_lr: float.
            The optimizer's lr at the BEGINNING of training, BEFORE any decay has taken place. This is constant over
            a training run.
        t: int.
            The current CUMULATIVE mini-batch number from the BEGINNING OF TRAINING (NOT the beginning of the epoch).
        num_steps: int.
            Total number of mini-batches that will be processed over the ENTIRETY of training (i.e. across ALL epochs).
        min_allowed_lr: float.
            We do not allow the linear lr decay to set the lr below this value.
            For example, if the lr decay equation says lr should be 0.001 but if min_allowed_lr=0.01, then we actually
            set the lr to 0.01.
        '''
        new_lr = initial_lr * (1 - ((t + 1) / num_steps))
        new_lr = max(float(min_allowed_lr), float(new_lr))
        self.opt.learning_rate.assign(new_lr)

    def get_word_embedding(self, wordind):
        '''Given the word index `wordind` retrieve and return the corresponding embedding vector.'''
        return self.embedding.get_wts()[wordind].numpy()

    def get_all_embeddings(self):
        '''Retrieve and return the embedding vectors for ALL words in the vocab.'''
        return self.embedding.get_wts().numpy()

    def get_bias(self):
        '''Retrieve and return the embedding layer bias.'''
        return self.embedding.get_b().numpy()

    def save_embeddings(self, path='export', filename='embeddings.npz'):
        '''Saves the embeddings to disk.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        path: str.
            Folder path where the embeddings should be saved.
        filename: str.
            Name of the file to which the embeddings should be saved. Should have a .npz file extension.
        '''
        full_path = os.path.join(path, filename)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        wts = self.get_all_embeddings()
        b = self.get_bias()
        np.savez_compressed(full_path, embeddings=wts, bias=b)

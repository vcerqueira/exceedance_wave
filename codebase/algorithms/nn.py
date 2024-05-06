import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


class DeepNet:

    def __init__(self, in_hre: bool, n_epochs: int, batch_size: int):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.method = 'FFNN'
        if in_hre:
            self.model_name = f'{self.method}_in_hre'
        else:
            self.model_name = f'{self.method}_no_hre'

        if self.method in ['FFNN']:
            self.is_recurrent_ = False
        else:
            self.is_recurrent_ = True

        self.model = None
        self.scaler = MinMaxScaler()

    def fit(self, X_train, y_train):

        k = X_train.shape[1]

        self.scaler.fit(X_train)

        X_train_scl = self.scaler.transform(X_train)

        if isinstance(X_train_scl, pd.DataFrame):
            X_train_scl = X_train_scl.values

        if self.is_recurrent_:
            X_train_scl = X_train_scl.reshape((X_train_scl.shape[0], X_train_scl.shape[1], 1))

        self.model = self.build_ff_model(shape=k,
                                         dropout=0.1,
                                         nunits_l1=64,
                                         nunits_l2=16,
                                         nunits_out=1,
                                         loss='mse')

        model_checkpoint_callback = ModelCheckpoint(
            filepath=f'output_nn/{self.model_name}',
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.model.fit(X_train_scl, y_train,
                       epochs=self.n_epochs,
                       batch_size=self.batch_size,
                       validation_split=0.1,
                       callbacks=[model_checkpoint_callback])

    def predict(self, X_test):

        X_test_scl = self.scaler.transform(X_test)

        if isinstance(X_test_scl, pd.DataFrame):
            X_test_scl = X_test_scl.values

        if self.is_recurrent_:
            X_test_scl = X_test_scl.reshape((X_test_scl.shape[0], X_test_scl.shape[1], 1))

        best_model = load_model(f'output_nn/{self.model_name}')

        yhat_ts = best_model.predict(X_test_scl)

        return yhat_ts.flatten()

    @staticmethod
    def build_ff_model(shape, dropout, nunits_l1=128, nunits_l2=64, nunits_out=1, loss="mse"):
        model = Sequential()
        model.add(Dense(nunits_l1, activation='relu', input_shape=(shape,)))
        model.add(Dropout(dropout))
        model.add(Dense(nunits_l2, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(nunits_out))

        model.compile(optimizer='adam', loss=loss)

        return model

from base_kam_model import BaseModel
from wearable_toolkit import DivideMaxScalar
import torch
import torch.nn as nn
from customized_logger import logger as logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, PHASE, SUBJECT_WEIGHT, SUBJECT_HEIGHT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler, StandardScaler

USE_GPU = True


class TianCNN(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        # self.conv1 = nn.Conv1d(x_dim, 30, kernel_size=3, stride=1)

        self.conv1 = nn.Conv2d(1, 50, kernel_size=(5, 1), stride=1)
        self.pooling1 = nn.MaxPool2d((221, 50))
        self.flatten = nn.Flatten()
        self.conv2output = nn.Linear(50, y_dim * 230)
        self.y_dim = y_dim

    def forward(self, sequence):
        sequence = sequence.unsqueeze(1)
        sequence = self.conv1(sequence)
        sequence = self.pooling1(sequence)
        sequence = self.flatten(sequence)
        output = self.conv2output(sequence)
        output = torch.reshape(output, (-1, 230, self.y_dim))
        return output


class WangRNN(nn.Module):
    def __init__(self, x_dim, y_dim, nlayer=2):
        super(WangRNN, self).__init__()
        hidden_dim = 10
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.y_dim = y_dim
        self.rnn_layer = nn.GRU(x_dim, hidden_dim, 2, dropout=0.5, batch_first=True, bidirectional=True)
        self.hidden2output = nn.Linear(2*hidden_dim, y_dim)
        for layer in [self.hidden2dense, self.dense2output]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, hidden, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.rnn_layer(sequence, hidden)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=230)
        output = self.hidden2output(gru_out)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayer*2, batch_size, self.hidden_dim)


class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=30, nlayer=2):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.rnn_layer = nn.LSTM(x_dim, hidden_dim, nlayer, dropout=0, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.hidden2dense = nn.Linear(2*hidden_dim, hidden_dim)
        self.dense2output = nn.Linear(hidden_dim, y_dim)
        for layer in [self.hidden2dense, self.dense2output]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, hidden, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.rnn_layer(sequence, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=230)
        relu_out = self.hidden2dense(lstm_out).clamp(min=0)
        output = self.dense2output(relu_out)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayer*2, batch_size, self.hidden_dim),
                weight.new_zeros(self.nlayer*2, batch_size, self.hidden_dim))


class TianModel(BaseModel):
    def __init__(self, data_path, x_fields, y_fields, weights, base_scalar):
        BaseModel.__init__(self, data_path, x_fields, y_fields, weights, base_scalar)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3

    @staticmethod
    def loss_fun_emphasize_peak(y_pred, y):
        peak_locs = torch.argmax(y, dim=1).reshape([-1])
        loss_peak = (y_pred[range(y.shape[0]), peak_locs] - y[range(y.shape[0]), peak_locs]).pow(2).mean()
        y_pred_non_zero = y_pred[y != 0]
        y_non_zero = y[y != 0]
        loss_profile = (y_pred_non_zero - y_non_zero).pow(2).mean()
        return (loss_profile + loss_peak) * 1e3

    @staticmethod
    def loss_fun_only_positive(y_pred, y):
        y_pred_positive = y_pred[y > 0]
        y_positive = y[y > 0]
        loss_positive = (y_pred_positive - y_positive).pow(2).sum()
        return loss_positive

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)
        x_train, y_train = np.concatenate(list(x_train.values()), axis=2), y_train['output']
        x_validation, y_validation = np.concatenate(list(x_validation.values()), axis=2), y_validation['output']
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        train_step_lens = torch.from_numpy(self.train_step_lens)
        nn_model = TianCNN(x_train.shape[2], y_train.shape[2])
        pytorch_total_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
        logging.info('Model has {} parameters.'.format(pytorch_total_params))

        if USE_GPU:
            # x_train = x_train.cuda()
            # y_train = y_train.cuda()
            nn_model = nn_model.cuda()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=2e-5, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(nn_model.parameters())

        batch_size = 20
        train_ds = TensorDataset(x_train, y_train, train_step_lens)
        train_size = int(0.98 * len(train_ds))
        vali_from_train_size = len(train_ds) - train_size
        train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds, [train_size, vali_from_train_size])
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=vali_from_train_size)

        x_validation = torch.from_numpy(x_validation).float()
        y_validation = torch.from_numpy(y_validation).float()
        if USE_GPU:
            x_validation = x_validation.cuda()
            y_validation = y_validation.cuda()
        vali_step_lens = torch.from_numpy(self.validation_step_lens)
        vali_from_test_ds = TensorDataset(x_validation, y_validation, vali_step_lens)
        num_of_step_for_peek = int(0.05 * len(x_validation))
        vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(x_validation) - num_of_step_for_peek])
        vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=len(vali_from_test_ds))

        logging.info('\tEpoch\t\tTrain_Loss\tVali_train_Loss\tVali_test_Loss\t\tDuration\t\t')
        for epoch in range(10):
            epoch_start_time = time.time()
            for i_batch, (xb, yb, lens) in enumerate(train_dl):
                if USE_GPU:
                    xb = xb.cuda()
                    yb = yb.cuda()

                optimizer.zero_grad()

                # # For RNN
                # hidden = nn_model.init_hidden(xb.shape[0])
                # y_pred, _ = nn_model(xb, hidden, lens)

                # For CNN
                y_pred = nn_model(xb)

                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
                train_loss = loss_fn(y_pred, yb)

                if epoch == 0 and i_batch == 0:
                    vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn, batch_size)
                    vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn, batch_size)
                    logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
                        epoch, train_loss.item(), vali_from_train_loss, vali_from_test_loss, time.time() - epoch_start_time))
                    # print(epoch, round(train_loss.item(), 2), round(0.0, 2), sep='\t\t')
                train_loss.backward()
                optimizer.step()

            vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn, batch_size)
            vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn, batch_size)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
                epoch, train_loss.item(), vali_from_train_loss.item(), vali_from_test_loss.item(), time.time() - epoch_start_time))
        return nn_model

    @staticmethod
    def evaluate_validation_set(nn_model, validation_dl, loss_fn, batch_size):
        for x_validation, y_validation, lens in validation_dl:
            if USE_GPU:
                x_validation = x_validation.cuda()
                y_validation = y_validation.cuda()
            # # For RNN
            # hidden = nn_model.init_hidden(x_validation.shape[0])
            # y_validation_pred, _ = nn_model(x_validation, hidden, lens)

            # For CNN
            y_validation_pred = nn_model(x_validation)

            validation_loss = loss_fn(y_validation_pred, y_validation) / len(y_validation) * batch_size
            return validation_loss

    def predict(self, nn_model, x_test):
        self.test_step_lens = self._get_step_len(x_test)
        x_test = np.concatenate(list(x_test.values()), axis=2)
        x_test = torch.from_numpy(x_test)
        if USE_GPU:
            x_test = x_test.cuda()
        # # For RNN
        # hidden = nn_model.init_hidden(x_test.shape[0])
        # y_pred, _ = nn_model(x_test, hidden, self.test_step_lens)

        # For CNN
        y_pred = nn_model(x_test)

        y_pred = y_pred.detach().cpu().numpy()
        return {'output': y_pred}

    @staticmethod
    def _get_step_len(data, input_cate='main_input_acc', feature_col_num=0):
        """

        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col_num: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[input_cate][:, :, feature_col_num]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=1)
        return data_len


if __name__ == "__main__":

    data_path = DATA_PATH + '/40samples+stance_swing+padding_zero.h5'
    IMU_FIELDS_ACC = ['AccelX', 'AccelY', 'AccelZ']
    IMU_FIELDS_GYR = ['GyroX', 'GyroY', 'GyroZ']
    IMU_DATA_FIELDS_ACC = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_ACC]
    IMU_DATA_FIELDS_GYR = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_GYR]
    video_cols = [loc + '_' + axis + '_' + angle for loc in ['RHip', 'LHip', 'RKnee', 'LKnee']
                  for angle in ['90', '180'] for axis in ['x', 'y']]
    output_cols = ['RIGHT_KNEE_ADDUCTION_MOMENT']

    x_fields = {'main_input_acc': IMU_DATA_FIELDS_ACC,
                'main_input_gyr': IMU_DATA_FIELDS_GYR,
                # 'main_input_vid': video_cols,
                'aux_input': [SUBJECT_WEIGHT, SUBJECT_HEIGHT]}
    y_fields = {'output': output_cols}
    weights = {'output': [PHASE]*len(output_cols)}

    model = TianModel(data_path, x_fields, y_fields, weights, DivideMaxScalar)
    model.cali_via_gravity()
    subjects = model.get_all_subjects()
    model.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    # model.cross_validation(subjects)
    plt.show()

from functools import partial
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.pytorch.model_ssl import ViViTSSL as GTSModel
from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time
import wandb
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import wandb_mixin, WandbLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="transformer", entity="aufl")

class GTSSupervisor:
    def __init__(self, save_adj_name, temperature, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        #self._mr = kwargs.get('mr')
        self._train_kwargs["mr"] = kwargs.get('mr')
        self._train_kwargs["mode"] = kwargs.get('mode')
        self.temperature = float(temperature)
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.save_adj_name = save_adj_name
        self.epoch_use_regularization = self._train_kwargs.get('epoch_use_regularization')
        self.num_sample = self._train_kwargs.get('num_sample')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        # self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        ### Feas
        if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
            df = pd.read_hdf('./data/metr-la.h5')
        elif self._data_kwargs['dataset_dir'] == 'data/PEMS-BAY':
            df = pd.read_hdf('./data/pems-bay.h5')
        #else:
        #    df = pd.read_csv('./data/pmu_normalized.csv', header=None)
        #    df = df.transpose()
        num_samples = df.shape[0]
        num_train = round(num_samples * 0.7)
        df = df[:num_train].values
        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)
        #print(self._train_feas.shape)

        k = self._train_kwargs.get('knn_k')
        knn_metric = 'cosine'
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        GTS_model = GTSModel(12,207,1, in_channels=2, mask_ratio=kwargs.get('mr'))
        self.GTS_model = GTS_model.cuda() if torch.cuda.is_available() else GTS_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'GTS_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.GTS_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.GTS_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.GTS_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        for k, v in kwargs.items():
            print(f"Training parameter {k}: {v}")
            setattr(wandb.config, k, v)
        '''
        config = {
            "mask_ratio": tune.choice([0.15, 0.3, 0.45, 0.6, 0.75]),
            "wandb": {
                "project": "transformer",
                "log_config": True,
                },
        }
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=kwargs["epochs"],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "training_iteration"])
        '''
        '''
        {'base_lr': 0.0015, 'dropout': 0, 'epoch': 0, 'epochs': 150, 'epsilon': 0.001, 'global_step': 0, 'lr_decay_ratio': 0.1, 'max_grad_norm': 5, 'max_to_keep': 100, 'min_learning_rate': 2e-06, 'optimizer': 'adam', 'patience': 100, 'steps': [20, 30, 40], 'test_every_n_epochs': 5, 'knn_k': 10, 'epoch_use_regularization': 150, 'num_sample': 10, 'save_model': 1}
        '''
        '''result = tune.run(
            partial(self._train, base_lr=kwargs["base_lr"],
                dropout=kwargs["dropout"],
                epoch=kwargs["epoch"],
                epochs=kwargs["epochs"],
                epsilon=kwargs["epsilon"],
                global_step=kwargs["global_step"],
                lr_decay_ratio=kwargs["lr_decay_ratio"],
                max_grad_norm=kwargs["max_grad_norm"],
                max_to_keep=kwargs["max_to_keep"],
                min_learning_rate=kwargs["min_learning_rate"],
                optimizer=kwargs["optimizer"],
                patience=kwargs["patience"],
                steps=kwargs["steps"],
                test_every_n_epochs=kwargs["test_every_n_epochs"],
                knn_k=kwargs["knn_k"],
                epoch_use_regularization=kwargs["epoch_use_regularization"],
                num_sample=kwargs["num_sample"],
                save_model=kwargs["save_model"]

                ),
            #resources_per_trial={"cpu": 2, "gpu": 2},
            config=config,
            #num_samples=100,
            #scheduler=scheduler,
            #progress_reporter=reporter
            )'''
        return self._train(**kwargs,config={"mask_ratio": kwargs.pop("mr", 0.3)})
    
    
    def evaluate(self,label, dataset='val', batches_seen=0, gumbel_soft=True, config=None):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            #rmses = []
            mses = []
            temp = self.temperature
            
            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []
            label = 'without_regularization'

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                loss, _, _ = self.GTS_model(x, mask_ratio=config["mask_ratio"])
                if label == 'without_regularization': 
                    # loss = self._compute_loss(y, output)
                    # y_true = self.standard_scaler.inverse_transform(y)
                    # y_pred = self.standard_scaler.inverse_transform(output)
                    # mapes.append(masked_mape_loss(y_pred, y_true).item())
                    # mses.append(masked_mse_loss(y_pred, y_true).item())
                    #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    losses.append(loss.item())
                    
                    
                    # Followed the DCRNN TensorFlow Implementation
                    # l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                    # m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                    # r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                    # l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                    # m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                    # r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                    # l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                    # m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                    # r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
                    

                else:
                    loss_1 = self._compute_loss(y, None)
                    # pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
                    # true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    # compute_loss = torch.nn.BCELoss()
                    # loss_g = compute_loss(pred, true_label)
                    # loss = loss_1 + loss_g
                    # # option
                    # # loss = loss_1 + 10*loss_g
                    # losses.append((loss_1.item()+loss_g.item()))

                    # y_true = self.standard_scaler.inverse_transform(y)
                    # y_pred = self.standard_scaler.inverse_transform(output)
                    # mapes.append(masked_mape_loss(y_pred, y_true).item())
                    # #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    # mses.append(masked_mse_loss(y_pred, y_true).item())
                    
                    # # Followed the DCRNN TensorFlow Implementation
                    # l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                    # m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                    # r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                    # l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                    # m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                    # r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                    # l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                    # m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                    # r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())

                #if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
            mean_loss = np.mean(losses)
            # mean_mape = np.mean(mapes)
            # mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option
            
            if dataset == 'test':
                
                # Followed the DCRNN PyTorch Implementation
                message = 'Test: mae: {:.4f}'.format(mean_loss)
                self._logger.info(message)
                
                # Followed the DCRNN TensorFlow Implementation
                # message = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                #                                                                            np.sqrt(np.mean(r_3)))
                # self._logger.info(message)
                # message = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                #                                                                            np.sqrt(np.mean(r_6)))
                # self._logger.info(message)
                # message = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                #                                                                            np.sqrt(np.mean(r_12)))
                # self._logger.info(message)

            # self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
            if label == 'without_regularization':
                return mean_loss
            else:
                return mean_loss

    
    def _train(self, config, base_lr,
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=0,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        val_steps = 0
        for epoch_num in range(self._epoch_num, epochs):
            print("Num of epoch:",epoch_num)
            self.GTS_model = self.GTS_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            temp = self.temperature
            gumbel_soft = True

            # if epoch_num < self.epoch_use_regularization:
            #     label = 'with_regularization'
            # else:
            #     label = 'without_regularization'
            label = 'without_regularization'


            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                wandb.watch(self.GTS_model)
                loss, _, _ = self.GTS_model(x, mask_ratio=config["mask_ratio"])
                # if (epoch_num % epochs) == epochs - 1:
                #     output = self.GTS_model(label, x, self._train_feas, temp, gumbel_soft, y, batches_seen)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

                self.GTS_model.to(device)
                
                #if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
                if label == 'without_regularization':  # or label == 'predictor':
                    losses.append(loss.item())
                else:
                    loss_1 = self._compute_loss(y, output)
                    pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
                    true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                    # option
                    # loss = loss_1 + 10*loss_g
                    losses.append((loss_1.item()+loss_g.item()))

                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.GTS_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()

            if label == 'without_regularization':
                wandb.log({'train_loss': np.mean(losses)})
                wandb.log({'epoch': epoch_num})
                val_loss = self.evaluate(label, dataset='val', batches_seen=batches_seen, gumbel_soft=gumbel_soft, config=config)
                val_steps += 1
                end_time2 = time.time()
                # self._writer.add_scalar('training loss',
                #                         np.mean(losses),
                #                         batches_seen)

                if (epoch_num % log_every) == log_every - 1:
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {}, val_rmse: {}, lr: {:.6f}, ' \
                              '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                        np.mean(losses), val_loss, "val_mape", "val_rmse",
                                                        lr_scheduler.get_lr()[0],
                                                        (end_time - start_time), (end_time2 - start_time))
                    self._logger.info(message)

                if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                    test_loss = self.evaluate(label, dataset='test', batches_seen=batches_seen, gumbel_soft=gumbel_soft, config=config)
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {}, test_rmse: {}, lr: {:.6f}, ' \
                              '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                        np.mean(losses), test_loss, "test_mape", "test_rmse",
                                                        lr_scheduler.get_lr()[0],
                                                        (end_time - start_time), (end_time2 - start_time))
                    self._logger.info(message)
            else:
                val_loss = self.evaluate(label, dataset='val', batches_seen=batches_seen, gumbel_soft=gumbel_soft, config=config)

                end_time2 = time.time()

                # self._writer.add_scalar('training loss', np.mean(losses), batches_seen)

                if (epoch_num % log_every) == log_every - 1:
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}'.format(epoch_num, epochs,
                                                                                             batches_seen,
                                                                                             np.mean(losses), val_loss)
                    self._logger.info(message)
                if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                    test_loss = self.evaluate(label, dataset='test', batches_seen=batches_seen, gumbel_soft=gumbel_soft, config=config)
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, lr: {:.6f}, ' \
                              '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                        np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                                        (end_time - start_time), (end_time2 - start_time))
                    self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            '''with tune.checkpoint_dir(epoch_num) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((self.GTS_model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=(val_loss / val_steps))'''
            wandb.log({'val_loss': val_loss})

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

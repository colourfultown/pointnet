#coding:utf-8
'''
 author:dabo
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os,random,time

import tf_util
from transform_nets import input_transform_net, feature_transform_net


class PointNetSeg():
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 2
        self.GPU_INDEX = 0
        self.feature_length = 3
        self.if_use_intensity = False
        self.label_dim = 8
        self.DECAY_STEP = 200000
        self.BASE_LEARNING_RATE = 0.001
        self.DECAY_RATE = 0.7
        self.BN_DECAY_DECAY_STEP = self.DECAY_STEP
        self.BN_DECAY_DECAY_RATE  = 0.5
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_CLIP = 0.99
        self.end_points = {}
        self.reg_weight = 0.0001
        self.MOMENTUM = 0.9
        self.OPTIMIZER = "adam"
        self.epoches = 2
        self.train_ratio = 0.9
        self.save_dir = "log/"

        self._print_ = None

        self.intersection_train = np.zeros(self.label_dim)
        self.union_train = np.ones(self.label_dim)
        self.intersection_valid = np.zeros(self.label_dim)
        self.union_valid = np.ones(self.label_dim)

        self.init_graph()
        self.init_session()

    def init_graph(self):
        # with tf.device('/gpu:'+str(self.GPU_INDEX)):
            # ---------- 定义输入，[batch_size, num_point, feature_length] ---------
            self.input_pointcloud = tf.placeholder(dtype=tf.float32, shape=(None, None, self.feature_length))
            self.ture_label = tf.placeholder(dtype=tf.int32, shape=(None, None))

            self._print_ = self.ture_label

            self.is_training = tf.placeholder(tf.bool, shape=())
            self.num_point = tf.placeholder(tf.int32)
            batch_size = self.input_pointcloud.get_shape()[0].value
            # num_point = self.input_pointcloud.get_shape()[1].value

            # ---------- 定义step，衰减计算， ----------
            step = tf.Variable(0)
            bn_decay = self.get_bn_decay(step)


            # ---------- 开始定义网络 ----------
            with tf.variable_scope('transform_net1') as sc:
                transform = input_transform_net(self.input_pointcloud, self.is_training, self.num_point, bn_decay, K=3)
            point_cloud_transformed = tf.matmul(self.input_pointcloud, transform)
            input_image = tf.expand_dims(point_cloud_transformed, -1)

            net = tf_util.conv2d(input_image, 64, [1, 3],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv2', bn_decay=bn_decay)

            with tf.variable_scope('transform_net2') as sc:
                transform = feature_transform_net(net, self.is_training, self.num_point, bn_decay, K=64)
            self.end_points['transform'] = transform
            net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
            point_feat = tf.expand_dims(net_transformed, [2])

            net = tf_util.conv2d(point_feat, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv3', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv4', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 1024, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv5', bn_decay=bn_decay)



            # global_feat = tf_util.max_pool2d(net, [self.num_point, 1],
            #                                  padding='VALID', scope='maxpool')
            # ----------  ----------
            global_feat = tf.reduce_max(net, axis=1, keepdims=True)




            global_feat_expand = tf.tile(global_feat, [1, self.num_point, 1, 1])
            concat_feat = tf.concat([point_feat, global_feat_expand], axis=3)

            net = tf_util.conv2d(concat_feat, 512, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv6', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv7', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv8', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=self.is_training,
                                 scope='conv9', bn_decay=bn_decay)

            net = tf_util.conv2d(net, self.label_dim, [1, 1],
                                 padding='VALID', stride=[1, 1], activation_fn=None,
                                 scope='conv10')
            self.pred = tf.squeeze(net, [2])  # BxNxC

            self.pre_class = tf.argmax(self.pred, 2)

            # ---------- 计算loss ----------
            self.loss = self.loss_function_balance_ce()

            # self.loss = self.loss_function_no0()

            # labels = tf.one_hot(self.ture_label, depth=self.label_dim, axis=2)
            # logits = self.pred
            # self.loss = self.focal_loss(labels, logits)

            # ---------- 计算metrics acc ----------
            self.acc = self.metrics_acc()

            # ---------- 定义优化方式，返回op ----------
            # learning_rate = self.get_learning_rate(step)

            learning_rate = self.learning_rate

            if self.OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.MOMENTUM)
            elif self.OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=step) 

            # --------- 保存模型的op-saver ----------

            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]


            bn_moving_vars += [g for g in g_list if 'ExponentialMovingAverage' in g.name]

            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3)


    def loss_function_balance_ce(self):
        x1 = tf.nn.softmax(self.pred, axis=2)

        x2 = tf.one_hot(self.ture_label, depth=self.label_dim, axis=2)
        # tf.clip_by_value(x1, 1.0, 1e-6)
        x3 = tf.log(x1+1E-8) * x2
        ce_loss = -tf.reduce_sum(x3, axis=2)# batchsize * pointnum
        neg_weight = tf.cast(tf.equal(self.ture_label, 0), tf.float32)# batchsize * pointnum
        pos_weight = 1 - neg_weight  # batchsize * pointnum

        n_neg = tf.reduce_sum(neg_weight)
        n_pos = tf.reduce_sum(pos_weight)

        def has_pos():
            return tf.reduce_sum(ce_loss * pos_weight) / n_pos
        def has_neg():
            return tf.reduce_sum(ce_loss * neg_weight) / n_neg
        def no():
            return tf.constant(0.0)


        pos_loss = tf.cond(n_pos > 0, has_pos, no)
        neg_loss = tf.cond(n_neg > 0, has_neg, no)

        loss = (pos_loss + neg_loss) / 2.0
        return loss

    def focal_loss(self, labels, logits, gamma=2, alpha=0.5, normalize=True):
        # ------- logits 是预测值 ---------
        labels = tf.reshape(labels, [-1, self.label_dim])
        logits = tf.reshape(logits, [-1, self.label_dim])


        labels = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
        labels = tf.cast(labels, tf.float32)
        probs = tf.sigmoid(logits)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        alpha_t = tf.ones_like(logits) * alpha
        alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
        probs_t = tf.where(labels > 0, probs, 1.0 - probs)

        focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
        loss = focal_matrix * ce_loss

        loss = tf.reduce_sum(loss)
        if normalize:
            n_pos = tf.reduce_sum(labels)
            total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
            total_weights = tf.Print(total_weights, [n_pos, total_weights])

            #         loss = loss / total_weights
            def has_pos():
                return loss / tf.cast(n_pos, tf.float32)

            def no_pos():
                # total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
                # return loss / total_weights
                return loss

            loss = tf.cond(n_pos > 0, has_pos, no_pos)
        return loss
    def metrics_acc(self):
        correct = tf.equal(tf.argmax(self.pred, 2), tf.to_int64(self.ture_label))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def loss_function_no0(self):
        labels = tf.one_hot(self.ture_label, depth=self.label_dim, axis=2)

        logits = tf.slice(self.pred, [0, 0, 1], [-1, -1, -1])
        labels = tf.slice(labels, [0, 0, 1], [-1, -1, -1])

        labels = tf.reshape(labels, [-1, self.label_dim-1])
        logits = tf.reshape(logits, [-1, self.label_dim-1])


        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        classify_loss = tf.reduce_mean(loss)
        # Enforce the transformation as orthogonal matrix
        transform = self.end_points['transform']  # BxKxK
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)
        loss_end = classify_loss + mat_diff_loss * self.reg_weight
        return loss_end
    def loss_function(self):

        labels = tf.reshape(self.ture_label, [-1])
        logits = tf.reshape(self.pred, [-1, self.label_dim])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        classify_loss = tf.reduce_mean(loss)
        # Enforce the transformation as orthogonal matrix
        transform = self.end_points['transform']  # BxKxK
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)
        loss_end = classify_loss + mat_diff_loss * self.reg_weight
        return loss_end



    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.BASE_LEARNING_RATE,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.DECAY_STEP,  # Decay step.
            self.DECAY_RATE,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            self.BN_INIT_DECAY,
            batch * self.batch_size,
            self.BN_DECAY_DECAY_STEP,
            self.BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.Session(graph=tf.get_default_graph(), config=config)


    def train(self, input_dir):
        print("start train")
        # with tf.device('/gpu:' + str(self.GPU_INDEX)):

        # ---------- 注意在：获取batch， 要把相同的num-point的图放在一起 ----------
        self.category_dir = os.path.join(input_dir, "category")
        self.intensity_dir = os.path.join(input_dir, "intensity")
        self.pts_dir = os.path.join(input_dir, "pts")
        file_names = os.listdir(self.category_dir)
        #file_names = file_names[0:100]
        num2file = self.files_sort_by_pointnum(self.pts_dir, file_names)
        print("end num2file")
        time_start = time.time()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                batch_file = self.generate_batch_files(num2file)

                haha = int(len(batch_file) * self.train_ratio)
                batch_file_train = batch_file[:haha]
                batch_file_valid = batch_file[haha:]

                for step, one_batch_file_list in enumerate(batch_file_train):

                    inputs, labels, num_point = self.generate_feeddict_data(one_batch_file_list)
                    # inputs 的维度：batch的数量 * batch-size * num—point * 特征数量(初始是3，3维坐标)
                    # labels 的维度：batch的数量 * batch-size * num—point

                    feed_dict = {
                        self.input_pointcloud: inputs,
                        self.ture_label: labels,
                        self.is_training: True,
                        self.num_point:num_point
                    }

                    # print__ = sess.run(self._print_, feed_dict)

                    _, _loss, _acc, _pre_class = sess.run([self.train_op, self.loss, self.acc, self.pre_class],
                                                          feed_dict=feed_dict)

                    print(step)
                    if step % 10 == 0:

                        # ---------- 对全部验证集合进行一次验证, 太耗时了，不行！修改到部分 5ci  ----------
                        _loss_valid_all = 0.0
                        _acc_valid_all = 0.0
                        _valid_data_length = 0
                        # random.shuffle(batch_file_valid)
                        for one_batch_file_list_valid in batch_file_valid[0:5]:
                            inputs_valid, labels_valid, num_point_valid =  self.generate_feeddict_data(one_batch_file_list_valid)
                            feed_dict_valid = {
                                self.input_pointcloud: inputs_valid,
                                self.ture_label: labels_valid,
                                self.is_training: False,
                                self.num_point:num_point_valid
                            }
                            _loss_valid, _acc_valid, _pre_class_valid = sess.run([self.loss, self.acc, self.pre_class],
                                                                                 feed_dict=feed_dict_valid)
                            _loss_valid_all += _loss_valid * len(inputs_valid)
                            _acc_valid_all += _acc_valid * len(inputs_valid)
                            _valid_data_length += len(inputs_valid)

                            # ----------每一个batch都做一次iou计算 ----------
                            valid_tmp_in, valid_tmp_un = self.iou_metrics(labels_valid, _pre_class_valid)
                            self.intersection_valid += valid_tmp_in
                            self.union_valid += valid_tmp_un


                        _loss_valid_all /= _valid_data_length
                        _acc_valid_all /= _valid_data_length



                        # ---------- 验证结束，打印信息 ----------
                        time_end = time.time()
                        print("epoch:%d/%d  step:%d/%d  time—interval:%.4f minutes"%(epoch+1, self.epoches, step+1, len(batch_file_train), (time_end-time_start)/60))
                        print("train_loss:%.6f  train_acc:%.6f"%(_loss, _acc))
                        print("valid_loss:%.6f  valid_acc:%.6f" % (_loss_valid_all, _acc_valid_all))

                        # ----------- 打印iou评价指标 ----------
                        train_tmp_in, train_tmp_un = self.iou_metrics(labels, _pre_class)
                        self.intersection_train += train_tmp_in
                        self.union_train += train_tmp_un

                        train_iou = np.mean(self.intersection_train[1:] / self.union_train[1:], dtype=float)
                        valid_iou = np.mean(self.intersection_valid[1:] / self.union_valid[1:], dtype=float)

                        print(self.intersection_train.astype(int), self.union_train.astype(int))
                        print(self.intersection_valid.astype(int), self.union_valid.astype(int))

                        self.intersection_train = np.zeros(self.label_dim)
                        self.union_train = np.ones(self.label_dim)
                        self.intersection_valid = np.zeros(self.label_dim)
                        self.union_valid = np.ones(self.label_dim)

                        print("train_iou:%.6f  valid_iou:%.6f" % (train_iou, valid_iou))

                        # ---------- 保存模型 ----------
                        if self.save_dir is not None:
                            if not os.path.exists(self.save_dir):
                                os.mkdir(self.save_dir)
                            save_path = os.path.join(self.save_dir, "model_epoch_%d_step_%d.ckpt"%(epoch, step))
                            self.saver.save(sess, save_path)



    def files_sort_by_pointnum(self, pts_dir, file_names):
        if os.path.exists("num2file.csv"):
            df = pd.read_csv("num2file.csv")
            df.columns = ['num', 'name']
            num2file = {num: list(df_['name'].values) for num, df_ in df.groupby("num")}
            return num2file
        num2file = {}

        df_new = pd.DataFrame()
        num = []
        name = []

        count = 0
        for i in file_names:
            if count %1000 == 0:
                print(count)
            count += 1
            path = os.path.join(pts_dir, i)
            df = pd.read_csv(path, header=None)
            num_point = len(df)
            num2file.setdefault(num_point,[]).append(i)

            num.append(num_point)
            name.append(i)

        df_new['num'] = num
        df_new['name'] = name
        df_new.to_csv("num2file.csv", index=False)

        return num2file
    def generate_batch_files(self, num2file):
        # ---------- 整个训练集的构建要shuffle 2次 ----------
        batch_file = [] # 返回结果，二维list，第二维是一个batch内用到的文件名
        for num, file_list in num2file.items():
            # random.shuffle(file_list)
            batches = np.ceil(float(len(file_list))/self.batch_size).astype(int)
            for i in range(batches):
                tmp = file_list[i * self.batch_size : min((i+1) * self.batch_size, len(file_list))]
                batch_file.append(tmp)
        # random.shuffle(batch_file)
        return batch_file
    def get_one_data(self, one_file_name):
        category_path = os.path.join(self.category_dir, one_file_name)
        intensity_path = os.path.join(self.intensity_dir, one_file_name)
        pts_path = os.path.join(self.pts_dir, one_file_name)


        if os.path.exists(category_path):
            label = pd.read_csv(category_path, header=None)
            label.columns = ['x']
            label = label['x'].values
        else:
            label = None


        feature = pd.read_csv(pts_path, header=None)
        feature = feature.values

        # ---------- 如果这个字段为True，使用强度信息，特征维度变成4 -----------
        if self.if_use_intensity:
            intensity = pd.read_csv(intensity_path, header=None)
            intensity = intensity.values
            feature = np.concatenate([feature, intensity], axis=1)

        return feature, label

    def generate_feeddict_data(self, one_batch_file_list):
        inputs = []  # batch的数量 * batch-size * num—point * 特征数量(初始是3，3维坐标)
        labels = []  # batch的数量 * batch-size * num—point
        for one_file_name in one_batch_file_list:
            inputs_one, labels_one = self.get_one_data(one_file_name)
            inputs.append(inputs_one)
            labels.append(labels_one)
        inputs = np.array(inputs)
        labels = np.array(labels)
        num_point = inputs.shape[1]
        return inputs, labels, num_point

    def predict(self, pathin, pathout, savedir=None):
        """
        预测函数
        :param pathin:  输入的路径
        :param pathout: 输出的路径
        :return:
        """
        print("predict start")
        if not os.path.exists(pathout):
            os.mkdir(pathout)
        self.category_dir = os.path.join(pathin, "category")
        self.intensity_dir = os.path.join(pathin, "intensity")
        self.pts_dir = os.path.join(pathin, "pts")
        file_names = os.listdir(self.pts_dir)
        print("test data count", len(file_names))
        time_start = time.time()
        with  self.session as sess:
            sess.run(tf.global_variables_initializer())
            if savedir is None:
                savedir = self.save_dir
            kpt = tf.train.latest_checkpoint(savedir)
            if kpt != None:
                self.saver.restore(sess, kpt)

            for index, one_file_name in enumerate(file_names):
                time_end = time.time()
                if index % 1000 == 0:
                    print("index:%d timeloss:%.4f hours "%(index, (time_end-time_start)/3600))
                fea, label = self.get_one_data(one_file_name)
                num_point = fea.shape[0]
                fea = np.expand_dims(fea, 0)
                feed_dict = {
                    self.input_pointcloud: fea,
                    self.is_training: False,
                    self.num_point: num_point
                }
                pred_label = sess.run(self.pred, feed_dict=feed_dict)
                pred_label = pred_label[0]
                pred_label = np.argmax(pred_label, axis=1)
                one_file_pathout = os.path.join(pathout, one_file_name)
                with open(one_file_pathout, 'w') as write_file:
                    for w in pred_label:
                        write_file.write(str(w) + "\n")


        print("predict over")

    def iou_metrics(self, true_label, pred_label, class_num = 0):
        """

        :param true_label: 一个banch内的数据
        :param pred_label:
        :param class_num:
        :return:
        """
        if class_num == 0 :
            class_num = self.label_dim
        intersection = np.zeros(class_num)
        union = np.zeros(class_num)
        for one_true, one_pred in zip(true_label, pred_label):
            for i, j in zip(one_true, one_pred):
                union[i] += 1
                union[j] += 1
                if i == j:
                    intersection[i] += 1
                    union[i] -= 1
        return intersection, union




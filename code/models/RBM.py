from .model import Model
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
	mean_absolute_error, mean_squared_error
from datetime import datetime


class RBM(Model):
    def _prepare_model(self):
        if "num_hidden_nodes" not in self.options:
            self.options["num_hidden_nodes"] = 15
        if "Gibbs_iters" not in self.options:
            self.options["Gibbs_iters"] = 20
        if "max_iters" not in self.options:
            self.options["max_iters"] = 5
        if "learning_rate" not in self.options:
            self.options["learning_rate"] = 0.1


    def add_train_data(self, df, response = "rating", features = ["user_id", "movie_id"]):
        self.df = df
        self.response = response
        self.feature1 = features[0]
        self.feature2 = features[1]

        F = self.options["num_hidden_nodes"]
        T = self.options["Gibbs_iters"]
        K = max(df[response])
        self.K = K
        m = max(df[self.feature2])
        lr = self.options["learning_rate"]

        df[self.feature1] = df[self.feature1] - 1  ##index start at 0 to help working with matrices
        df[self.feature2] = df[self.feature2] - 1

        list_feature1 = df[self.feature1].unique()
        list_feature2 = df[self.feature2].unique()



        r = np.asarray([sum(df[self.feature2] == i) for i in range(m)])  ## r[i]: number movie (i+1) is rated
        ra = np.zeros((m, K))  ## ra[i,k]: num movie i is rated k
        bh = np.zeros((F))  ## initialize bias for hidden unit
        bv = np.zeros((m, K))  ## initialize bias for visible rating
        for i in list_feature2:
            for k in range(K):
                ra[i, k] = np.sum(np.asarray(df[self.feature2] == i) & np.asarray(df[response] == k+1)) / np.sum(np.asarray(df[self.feature2] == i))    ## ra[i,k]: freq movie i is rated k
                bv[i, k] = np.log((ra[i,k] + 0.1) / (1 - ra[i,k]))
        W = np.random.normal(loc = 0.0, scale = 0.1, size = (m, F, K)) ## initialize weight
        #bv = bv - 4




        ## Idea: For each user, use Maximum likelihood to update W, bv, bh

        ## extract movie and rating for each user
        mu = [None] * len(list_feature1)    ## list of  movies rated by an user
        vi = [None] * len(list_feature1)    ## one-hot representation of rating of each user
        ratingId = [None] * len(list_feature1)     ## rating Id of (user, movie): convenience

        for u in list_feature1:

            mu[u] = list(df[df[self.feature1] == u][self.feature2])        ## list of  movies rated by an user
            vi[u] = np.zeros((len(mu[u]), K))
            for i in range(len(mu[u])):
                rate_k = list(df[np.asarray(df[self.feature1] == u) & np.asarray(df[self.feature2] == mu[u][i])][response])[0] - 1
                vi[u][i, rate_k] = 1                             ## V_[mu[u][i]]^k of user u
        self.mu = mu
        self.vi = vi


        # For update the parameters by getting the gradient from each user and do gradient ascent
        # with the average gradient from all users
        def update_para(W, bv, bh):
            dbh = np.zeros(bh.shape)
            dbv = np.zeros(bv.shape)
            dW = np.zeros(W.shape)
            for u in list_feature1:

                Wu = W[mu[u], :, :]  ##Weight for user u
                bvu = bv[mu[u], :]   ##bias for visible nodes of user u
                vu = vi[u]   ##visible rating for user u
                ## get the gradient for w, bv and bh using Contrastive Divergence (kind of Gibbs sampler)
                pv = np.zeros((len(mu[u]), K))

                ph_data = np.asarray([1 / (1 + np.exp(- bh[j] - np.dot(vu.flatten(), W[mu[u],j,:].flatten())))
                      for j in range(F)])
                h_data = np.asarray([np.random.choice(np.arange(2), 1, p=[1-ph_data[j], ph_data[j]])[0] for j in range(F)])
                h = h_data

                vp = vu  # stochastic version of v
                for _ in range(T):
                    for i in range(len(mu[u])):
                        # pv[i, :] = np.asarray([np.exp(bvu[i, k] + h.reshape(1, F) @ Wu[i,:,k].reshape(F,1)) for k in range(K)])
                        pv[i, :] = np.exp(bvu[i, :] + h.reshape(1, F) @ Wu[i, :, :].reshape(F, K))
                        pv[i, :] = pv[i, :] / sum(pv[i, :])
                        k_sample = np.random.choice(np.arange(K), 1, p=pv[i, :])[0]
                        vp[i, :] = np.zeros((K))
                        vp[i, k_sample] = 1

                    ph = np.asarray([1 / (1 + np.exp(- bh[j] - np.dot(vp.flatten(), W[mu[u], j, :].flatten())))
                        for j in range(F)])
                    h = np.asarray([np.random.choice(np.arange(2), 1, p=[1 - ph[j], ph[j]])[0] for j in range(F)])

                # contribution of each user to gradient
                # udbh = 1 / len(list_feature1) * (ph_data - ph)
                udbh = 1 / len(list_feature1) * (ph_data - h)
                udbv = vu - pv
                udW = np.zeros((len(mu[u]), F, K))
                for i in range(len(mu[u])):
                    udW[i, :, :] = 1/(r[mu[u][i]]) * (h_data.reshape(F, 1) @ (vu[i, :]).reshape(1, K) - ph.reshape(F, 1) @ (vp[i, :]).reshape(1, K))
                    udbv[i, :] = 1/(r[mu[u][i]] * K) * udbv[i, :]

                dbh += udbh
                dbv[mu[u], :] += udbv
                dW[mu[u], :, :] += udW

            # update parameter
            W += lr * dW
            bv += lr * dbv
            bh += lr * dbh
            return W, bv, bh

        def total_free_energy(W, bv, bh):
            list_free_energy = []
<<<<<<< HEAD
            #for u in list_feature1:
=======
            # for u in list_feature1:
>>>>>>> 20423e0f1f51fcd83fa005b6dd4fb911b075499d


            free_energy = np.mean(list_free_energy)
            return free_energy



        for _ in range(self.options["max_iters"]):
            print(_)
            W, bv, bh = update_para(W, bv, bh)

        self.W = W
        self.bh = bh
        self.bv = bv
        print('W is', W)
        print('bh is ', bh)
        print('bv is', bv)


    def train(self):
        mu = self.mu
        vi = self.vi
        W = self.W
        bh = self.bh
        bv = self.bv
        K = self.K
        F = self.options["num_hidden_nodes"]

        list_feature1 = self.df[self.feature1].unique()
        ##predict on train set
        self.df['pred_rating'] = '0'
        #self.df['pred_rating_hard'] = '0'
        for u in list_feature1:
            common_term = (vi[u].reshape(1, len(mu[u]) * K) @ np.swapaxes(W[mu[u], :, :], 1, 2).reshape(len(mu[u]) * K,
                                                                                                        F)).squeeze()  ## common term of shape (F,)
            print(common_term)
            for q in range(len(mu[u])):
                ## calculate P(vi_{mu[u][q]}^{k} = 1) for all k, then take expectation with respect to k to have the prediction
                missing_term = (W[mu[u][q], :, :].reshape(F, K) @ vi[u][q, :].reshape(K, 1)).squeeze()
                prob = [np.exp(bv[mu[u][q], k]) * np.prod(
                    np.exp(-common_term) + np.exp(-missing_term + W[mu[u][q], :, k].squeeze() + bh)) for k in range(K)]

                prob = prob / np.sum(prob)  ## normalization for soft prediction
                predict = np.dot(prob, np.arange(0, K))  ##expectation
                mov_index = self.df[(self.df[self.feature1] == u) & (self.df[self.feature2] == mu[u][q])].index.tolist()[0]
                self.df.at[mov_index, 'pred_rating'] = predict + 1

                #hard_predict = np.argmax(prob)
                #self.df.at[mov_index, 'pred_rating_hard'] = hard_predict + 1



        y_train = self.df[self.response]
        y_train_pred = self.df['pred_rating']
        #y_train_hard_pred = self.df['pred_rating_hard']


        y_train_pred_class = y_train_pred.apply(round)

        # compute metrics
        acc = accuracy_score(y_train, y_train_pred_class)
        prec = precision_score(y_train, y_train_pred_class, average="weighted", zero_division=0)
        rec = recall_score(y_train, y_train_pred_class, average="weighted", zero_division=0)
        mae = mean_absolute_error(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)


        self.metrics.update({
            "train_accuracy": acc,
            "train_precision": prec,
            "train_recall": rec,
            "train_mae": mae,
            "train_mse": mse
        })
        print(self.metrics)
        print(self.df)

    def test(self, test_df):
        F = self.options["num_hidden_nodes"]
        test_df[self.feature1] = test_df[self.feature1] - 1  ##index start at 0 to help working with matrices
        test_df[self.feature2] = test_df[self.feature2] - 1

        test_list_feature1 = test_df[self.feature1].unique()

        test_mu = [None] * len(test_list_feature1)  ## list of  movies rated by an user in test set
        test_vi = [None] * len(test_list_feature1)

        for u in test_list_feature1:

            test_mu[u] = list(test_df[test_df[self.feature1] == u][self.feature2])        ## list of  movies rated by an user
            test_vi[u] = np.zeros((len(test_mu[u]), self.K))
            for i in range(len(test_mu[u])):
                rate_k = list(test_df[np.asarray(test_df[self.feature1] == u) & np.asarray(test_df[self.feature2] == test_mu[u][i])][self.response])[0] - 1
                test_vi[u][i, rate_k] = 1                             ## V_[mu[u][i]]^k of user u

        test_df['pred_rating'] = '0'
        #test_df['pred_rating_hard'] = '0'
        for u in test_list_feature1:
            ## common term of shape (F,)
            # common_term = (self.vi[u].reshape(1, len(self.mu[u]) * self.K) @ np.swapaxes(self.W[self.mu[u], :, :], 1, 2).reshape(len(self.mu[u]) * self.K,
            #                                                                                             F)).squeeze()
            common_term = np.asarray([np.sum(np.multiply(self.vi[u], self.W[self.mu[u], j, :])) for j in range(F)])
            for q in range(len(test_mu[u])):
                ## calculate P(vi_{mu[u][q]}^{k} = 1) for all k, then take expectation with respect to k to have the prediction

                prob = [np.exp(self.bv[test_mu[u][q], k]) * np.prod(
                    1 + np.exp(common_term + self.W[test_mu[u][q], :, k].squeeze() + self.bh)) for k in range(self.K)]

                prob = prob / np.sum(prob)                      ##normalization for soft prediction
                predict = np.dot(prob, np.arange(0, self.K))    ##expectation
                mov_index = test_df[(test_df[self.feature1] == u) & (test_df[self.feature2] == test_mu[u][q])].index.tolist()[0]
                test_df.at[mov_index, 'pred_rating'] = predict + 1

                # hard_predict = int(np.argmax(prob))                ## hard prediction
                #
                # test_df.at[mov_index, 'pred_rating_hard'] = hard_predict + 1

        y_test = test_df[self.response]
        y_test_pred = test_df['pred_rating']
        #y_test_pred_hard = test_df['pred_rating_hard']

        y_test_pred_class = y_test_pred.apply(round)

        # compute metrics
        acc = accuracy_score(y_test, y_test_pred_class)
        prec = precision_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_test_pred_class, average="weighted", zero_division=0)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)

        self.metrics.update({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_mae": mae,
            "test_mse": mse
        })

        print(self.metrics)
        print(test_df)


    def log(self, path="models/log/rbm.tsv"):
        with open(path, "a") as log_file:
            entries = [
                str(datetime.now()),
                "RBM ({})".format(", ".join(["{}: {}".format(k, v) for k, v in self.options.items()])),
            ]
            entries.extend([str(val) for val in self.metrics.values()])
            log_file.write(
                "\n" + "\t".join(entries)
            )

















    ### Idea: Collect update from all users then update all parameters at once
        # for u in list_feature1:
        #     mu = df[df[feature1 == u]][feature2].tolist  ## movies rated by an user
        #     for i in range(len(mu)):
        #         v[i, df[df[feature1 == mu[i]] and df[features] == i][response][0]] = 1
        #
        #     ## get the gradient for w, bv and bh using Contrastive Divergence (kind of Gibbs sampler)
        #
        #     ph = np.zeros((F))
        #     pv = np.zeros((m, K))
        #
        #     ph_data = np.array(
        #         [1 / (1 + np.exp(- bh[j] - np.transpose(v.flatten()) @ np.array([W[i][j][:] for i in mu]).flatten()))
        #          for j in range(F)])
        #     ph = ph_data
        #
        #     vp = v  ## stochastic version of v
        #     for _ in range(T):
        #         h = np.array([np.random.choice(np.arange(2), 1, p=[1 - ph[j], ph[j]])[0] for j in range(F)])
        #         for i in range(len(mu)):
        #             pv[i, :] = np.array(
        #                 [np.exp(bv[i, k] + np.transpose(h) @ np.array([W[mu[i]][j][k] for j in range(F)])) for k in
        #                  range(K)])
        #             pv[i, :] = pv[i,] / sum(pv[i,])
        #             k_sample = np.random.choice(np.arange(K), 1, p=pv)[0]
        #             vp[i, :] = np.zeros((K))
        #             vp[i, k_sample] = 1
        #
        #         ph = [1 / (1 + np.exp(
        #             - bh[j] - np.transpose(vp.flatten()) @ np.array([W[i][j][:] for i in mu]).flatten()))
        #               for j in range(F)]
        #
        #     ## contribution of each user to gradient
        #     dbh += [((ph_data[j] - ph[j])) for j in range(F)]  ##estimate expectation by ph[j] not really exact??
        #     for i in range(len(mu)):
        #         dbv[mu[i], :] += 1 / len(mu) * (v[mu[i], :] - pv[i, :])
        #         dW[mu[i], :, :] += 1 / len(mu) * (ph_data @ np.transpose(v[i, :]) - h @ np.transpose(vp[i, :]))
        #
        # ## update parameter
        #
        # def update_para(u, W, bh, bv):
        #
        #     Wu = W[mu[u], :, :]  ##Weight for user u
        #     bvu = bv[mu[u], :]  ##bias for visible nodes of user u
        #     vu = vi[u]  ##visible rating for user u
        #     ## get the gradient for w, bv and bh using Contrastive Divergence (kind of Gibbs sampler)
        #     pv = np.zeros((len(mu[u]), K))
        #
        #     ph_data = np.asarray([1 / (1 + np.exp(- bh[j] - np.dot(vu.flatten(), W[mu[u], j, :].flatten())))
        #                           for j in range(F)])
        #     ph = ph_data
        #
        #     vp = vu  # stochastic version of v
        #     for _ in range(T):
        #         h = np.asarray([np.random.choice(np.arange(2), 1, p=[1 - ph[j], ph[j]])[0] for j in range(F)])
        #         for i in range(len(mu[u])):
        #             # pv[i, :] = np.asarray([np.exp(bvu[i, k] + h.reshape(1, F) @ Wu[i,:,k].reshape(F,1)) for k in range(K)])
        #             pv[i, :] = np.exp(bvu[i, :] + h.reshape(1, F) @ Wu[i, :, :].reshape(F, K))
        #             pv[i, :] = pv[i, :] / sum(pv[i, :])
        #             k_sample = np.random.choice(np.arange(K), 1, p=pv[i, :])[0]
        #             vp[i, :] = np.zeros((K))
        #             vp[i, k_sample] = 1
        #
        #         ph = [1 / (1 + np.exp(- bh[j] - np.dot(vp.flatten(), W[mu[u], j, :].flatten())))
        #               for j in range(F)]
        #
        #     # contribution of each user to gradient
        #     dbh = 1 / len(list_feature1) * (ph_data - ph)
        #     dbv = vu - pv
        #     dW = np.zeros((len(mu[u]), F, K))
        #     for i in range(len(mu[u])):
        #         dW[i, :, :] = 1 / r[mu[u][i]] * (
        #                     ph_data.reshape(F, 1) @ (vu[i, :]).reshape(1, K) - h.reshape(F, 1) @ (vp[i, :]).reshape(1,
        #                                                                                                             K))
        #         dbv[i, :] = 1 / r[mu[u][i]] * dbv[i, :]
        #         # update parameter
        #
        #     W[mu[u], :, :] += lr * dW
        #     bv[mu[u], :] += lr * dbv
        #     bh += lr * dbh
        #
        # for _ in range(self.options["max_iters"]):
        #     print(_)
        #     for u in list_feature1:
        #         update_para(u, W, bh, bv)
        #
        # self.W = W
        # self.bh = bh
        # self.bv = bv
        # print('W is', W)
        # print('bh is ', bh)
        # print('bv is', bv)
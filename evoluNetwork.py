import os
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def get_pca_tranformer(feat, n=64):
    pca = TruncatedSVD(n_components=n)
    pca.fit(feat)
    return pca


class EvolutionGraph:
    def __init__(self, feat, k, batchsize):
        self.k = k
        
        self.pca = get_pca_tranformer(feat)
        self.feat = self.pca.transform(feat)

        self.get_adj(batchsize)
    
    def _get_pca_tranformer(self, n=64):
        pca = TruncatedSVD(n_components=n)
        pca.fit(feat)
        return pca
    
    def get_adj(self, batch_size=5000):
        feat_t = self.feat.T
        slides = []
        # step1
        mat = self.feat[:batch_size]
        mat = mat.dot(feat_t)
        mat = self._get_neighs_init(mat, batch_size)
        slides.append(mat)
        # step2
        k = batch_size
        while (k+batch_size) < self.feat.shape[0]:
            mat = self.feat[k: k+batch_size]
            mat = mat.dot(feat_t)
            mat = self._get_neighs(mat, k)
            slides.append(mat)
            k += batch_size
            print("{} app handled".format(k))
        mat = self.feat[k:]
        mat = mat.dot(feat_t)
        mat = self._get_neighs(mat, k)
        slides.append(mat)

        self.adj = sp.vstack(slides)

    def _get_neighs(self, mat, start_id):
        print(mat.shape)
        data = []
        row = []
        col = []
        for iloc, d in enumerate(mat):
            d = d.squeeze()
            d[iloc+start_id:] = 0
            inds = np.argpartition(d, -self.k)[-self.k:]

            for ind in inds:
                if d[ind]:
                    data.append(d[ind])
                    row.append(iloc)
                    col.append(ind)
        coo_mat = sp.coo_matrix((data, (row, col)), shape=mat.shape)
        return coo_mat.tocsr()

    def _get_neighs_init(self, mat, threshold):
        print(mat.shape)
        data = []
        row = []
        col = []
        for iloc, d in enumerate(mat):
            d = d.squeeze()
            d[threshold:] = 0
            inds = np.argpartition(d, -self.k)[-self.k:]

            for ind in inds:
                if d[ind]:
                    data.append(d[ind])
                    row.append(iloc)
                    col.append(ind)
        coo_mat = sp.coo_matrix((data, (row, col)), shape=mat.shape)
        return coo_mat.tocsr()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="construct knn.")
    parser.add_argument('--keyword', type=str, default='drebin')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    data_dir = "" # set before runnig
    feat = sp.load_npz(os.path.join(data_dir, "{}_feat.npz".format(args.keyword)))
    graph = EvolutionGraph(feat, args.k, batchsize=10000)
    print(args.keyword, args.k, graph.adj)
    sp.save_npz(os.path.join(data_dir, "{}_knn_{}.npz".format(args.keyword, args.k)), graph.adj)

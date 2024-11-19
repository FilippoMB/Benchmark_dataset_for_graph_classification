from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import networkx as nx
from sklearn.preprocessing import OneHotEncoder


def make_instance(
        moon_c=[0,0],
        gaussq_c=[2,-2],
        blob_c=[[4,0]],
        n_nodes=90,
        random_n_nodes=False,
        cov=.2,
        k_neigh=5,
        conn_mode='connectivity',
        noisy_coords=False,
        noise_level=.1,
        plot_on=False):
    
    if random_n_nodes:
        n_nodes = np.random.randint(low=n_nodes//2, high=n_nodes)
    
    n_samples = n_nodes//3
    
    # initial features
    Xm, Ym = datasets.make_moons(n_samples=n_samples, noise=0.1)
    Xm[:,0] += moon_c[0]
    Xm[:,1] += moon_c[1]
    Xq, Yq = datasets.make_gaussian_quantiles(n_samples=n_samples, mean=gaussq_c, n_classes=2, cov=cov)
    Yq += 2
    Xb, Yb = datasets.make_blobs(n_samples=n_samples, centers=blob_c, cluster_std=cov*2)
    Yb += 4
    X = np.concatenate((Xm, Xq, Xb))
    X /= np.max(X,axis=0) 
    Y = np.concatenate((Ym, Yq, Yb))
    
    if plot_on:
        plt.scatter(X[:,0], X[:,1], c=Y)
        plt.title('initial features')
        plt.show()
    
    # build graph
    A = kneighbors_graph(X, n_neighbors=k_neigh, mode=conn_mode).todense()
    A = np.asarray(A)
    A = np.maximum(A, A.T)
    A /= A.max()  # normalize in [0,1]
    A = sp.csr_matrix(A, dtype=np.float32)
    G = nx.from_scipy_sparse_matrix(A)
    
    if plot_on:
        nx.draw_networkx(G, pos=nx.fruchterman_reingold_layout(G), with_labels=False, node_size=20, edge_color='lightgray', node_color=Y,
                 linewidths=1)
        plt.title('graph')
        plt.show()
        
    # node features
    F = OneHotEncoder(sparse=False, categories='auto').fit_transform(Y[...,None])
    
    if noisy_coords:
        X = np.tanh(X*1.1)
        X = np.multiply(X, np.diag(np.random.randn(X.shape[0],1)*noise_level))
        
        if plot_on:
            plt.scatter(X[:,0], X[:,1], c=Y)
            plt.title('noisy coords features')
            plt.show()
            
        F = np.concatenate((F, X), axis=-1)
        
    F_tuple = tuple(map(tuple, F))
    nx.set_node_attributes(G, dict(enumerate(F_tuple)), 'features')    
    
    return F.astype(np.float32), A, G


def make_dataset(
        moon =    [ [[4,0]],  [[2,-2]],  [[0,0]] ], #[ [[4,0],   [0,0]],     [[2,-2],[4,0]],      [[2,-2], [0,0]] ],           
        gaussq =  [ [[0,0]],  [[4,0]],  [[2,-2]] ], #[ [[0,0],   [4,0]],     [[0,0], [2,-2]],     [[4,0],  [2,-2]] ],          
        blob =    [ [[[2,-2]]],  [[[0,0]]],  [[[4,0]]] ], #[ [[[2,-2]],[[2,-2]]],  [[[4,0]], [[0,0]]],  [[[0,0]],  [[4,0]]] ],       
        n_nodes=90,
        random_n_nodes=False,
        cov=.2,
        k_neigh=5,
        conn_mode='connectivity',
        noisy_coords=False,
        noise_level=.5,
        plot_on=False,
        tr_size = 0.9,
        samples_per_subclass=150
        ):
    n_classes = len(moon)
    n_subclass = len(moon[0])
    print('n_classes:',n_classes,', n_subclasses:',n_subclass)
    
    tr_F = []
    tr_A = []
    tr_G = []
    tr_C = []
    
    val_F = []
    val_A = []
    val_G = []
    val_C = []
    
    te_F = []
    te_A = []
    te_G = []
    te_C = []
    
    for c in range(n_classes):
        for s in range(n_subclass):
            
            for _ in range(samples_per_subclass):
                F, A, G = make_instance(moon_c=moon[c][s],
                                       gaussq_c=gaussq[c][s],
                                       blob_c=blob[c][s],
                                       n_nodes=n_nodes,
                                       random_n_nodes=random_n_nodes,
                                       cov=cov,
                                       k_neigh=k_neigh,
                                       conn_mode=conn_mode,
                                       noisy_coords=noisy_coords,
                                       noise_level=noise_level,
                                       plot_on=plot_on)
                
                if np.random.rand() < tr_size:
                    if np.random.rand() < tr_size:
                        tr_F.append(F)
                        tr_A.append(A)
                        tr_G.append(G)
                        tr_C.append(c)
                    else:
                        val_F.append(F)
                        val_A.append(A)
                        val_G.append(G)
                        val_C.append(c) 
                else:
                    te_F.append(F)
                    te_A.append(A)
                    te_G.append(G)
                    te_C.append(c)
                    
    # one-hot class labels
    tr_C = np.asarray(tr_C)
    tr_C = OneHotEncoder(sparse=False, categories='auto').fit_transform(tr_C[...,None])
    val_C = np.asarray(val_C)
    val_C = OneHotEncoder(sparse=False, categories='auto').fit_transform(val_C[...,None])
    te_C = np.asarray(te_C)
    te_C = OneHotEncoder(sparse=False, categories='auto').fit_transform(te_C[...,None])
    
    return tr_F, tr_A, tr_G, tr_C.astype(np.float32), \
           val_F, val_A, val_G, val_C.astype(np.float32), \
           te_F, te_A, te_G, te_C.astype(np.float32) 
        

if __name__=='__main__':

    ds = 'hard_small'
    
    ds_kwargs = {
        'hard_small': {'n_nodes':80, 'cov':.2, 'k_neigh':3, 'samples_per_subclass':100},
        'hard_normal': {'n_nodes':200, 'cov':.2, 'k_neigh':3, 'samples_per_subclass':600},
        'easy_small': {'n_nodes':80, 'cov':.4, 'k_neigh':5, 'samples_per_subclass':100},
        'easy_normal': {'n_nodes':200, 'cov':.4, 'k_neigh':5, 'samples_per_subclass':600},
    }
    
    F, A, G = make_instance(
            n_nodes=80,
            cov=.4, 
            k_neigh=5, 
            conn_mode='connectivity',
            plot_on=True)
    
    tr_F, tr_A, tr_G, tr_C, val_F, val_A, val_G, val_C, te_F, te_A, te_G, te_C = make_dataset(
            n_nodes=ds_kwargs[ds]['n_nodes'],
            random_n_nodes=True,
            cov=ds_kwargs[ds]['cov'],
            k_neigh=ds_kwargs[ds]['k_neigh'],
            noisy_coords=False,
            conn_mode='connectivity',
            tr_size = 0.9,
            samples_per_subclass=ds_kwargs[ds]['samples_per_subclass']
            )
    
    np.savez(ds, 
         tr_adj=tr_A, 
         tr_feat=tr_F,
         tr_class=tr_C,
         val_adj=val_A, 
         val_feat=val_F,
         val_class=val_C,
         te_adj=te_A,
         te_feat=te_F,
         te_class=te_C)
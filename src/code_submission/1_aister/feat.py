import pandas as pd
import numpy as np
from util import timeclass,get_logger
from speed_up import sparse_dot
import networkx as nx

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class Feat:
    def __init__(self):
        pass
    
    @timeclass('Feat')
    def fit_transform_first(self,table):
#        self.reverse_decode_value_counts(table)
        pass
    
    @timeclass('Feat')
    def fit_transform(self,table,drop_sum_columns):
        degree_columns = self.degree(table)
        degree_bins_columns = self.degree_bins(table)
#        edge_weight_bins_columns = self.edge_weight_bins(table)
#        connected_subgraph_columns = self.connected_subgraph(table)
        neighbor_columns = self.get_neighbor(table)
        bin_2_neighbor_mean_degree_bins_columns = self.bin_2_neighbor_mean_degree_bins(table)
        
#        处理lgb用的feature
#        lgb_append = [degree_columns,degree_bins_columns,connected_subgraph_columns,neighbor_columns]
#        table.lgb_columns = table.lgb_columns.append(lgb_append)
        
#        gnn_append = []
        
        gnn_append = [degree_bins_columns,bin_2_neighbor_mean_degree_bins_columns]
        
#        elif not table.undirected_graph:
#            LOGGER.info('!!!edge_weight_bins_columns feature!!!')
#            gnn_append = [edge_weight_bins_columns]
        
#        gnn_append.append(drop_sum_columns)
        
        #sage用的feature
        table.sage_columns = table.sage_columns.append(gnn_append)
        
        #gat用的feature
        table.gat_columns = table.gat_columns.append(gnn_append)
        
        #tagc用的feature
        table.tagc_columns = table.tagc_columns.append(gnn_append)

        #gcn用的feature
        table.gcn_columns = table.gcn_columns.append(gnn_append)
        
    @timeclass('Feat')
    def degree(self,table):
        old_columns = table.df.columns
        df = table.df
        df_edge = table.df_edge
        
        if table.undirected_graph:
            LOGGER.info('Undirected graph')
            df['degree'] = df['node_index'].map(df_edge.groupby('src_idx').size().to_dict())
            df['degree'].fillna(0,inplace=True)
        else:
            LOGGER.info('Directed graph')
            df['out_degree'] = df['node_index'].map(df_edge.groupby('src_idx').size().to_dict())
            df['in_degree'] = df['node_index'].map(df_edge.groupby('dst_idx').size().to_dict())
            df['out_degree'].fillna(0,inplace=True)
            df['in_degree'].fillna(0,inplace=True)
            df['degree'] = df['out_degree']+df['in_degree']
            
            df['equal_out_in'] = 0
            df['out_more_than_in'] = 0
            df['in_more_than_out'] = 0
            df.loc[df['out_degree'] == df['in_degree'],'equal_out_in'] = 1
            df.loc[df['out_degree'] > df['in_degree'],'out_more_than_in'] = 1
            df.loc[df['out_degree'] < df['in_degree'],'in_more_than_out'] = 1
            
            LOGGER.info(f"out degree == in degree\n{df.loc[df['equal_out_in']==1,'label'].value_counts()}")
            LOGGER.info(f"out degree > in degree\n{df.loc[df['out_more_than_in']==1,'label'].value_counts()}")
            LOGGER.info(f"out degree < in degree\n{df.loc[df['in_more_than_out']==1,'label'].value_counts()}")
            
            #这里对特殊的数据集做处理，特重要
            if df.loc[df['out_degree'] <= df['in_degree'],'label'].nunique()==1:
                LOGGER.info(f'mask especial unbalanced data')
                table.especial = True
                #不训练这部分数据
                mask_index = df.loc[df['out_degree'] <= df['in_degree']].index
                table.directed_mask[mask_index] = 0
                
                df_tmp = df.loc[table.directed_mask.tolist()]
                LOGGER.info(f'after mask, {df_tmp.shape[0]} points, train:test = {(df_tmp["is_test"]==0).sum()}:{(df_tmp["is_test"]==1).sum()}')
                #不使用这些边
                df_edge = table.df_edge
                df_edge = df_edge[(df_edge['src_idx'].isin(df_tmp['node_index']))].reset_index(drop=True)
                LOGGER.info(f'after mask, {df_edge.shape[0]} edges')
                
                df_edge2 = df_edge.copy()
                df_edge2.rename(columns={'src_idx':'dst_idx','dst_idx':'src_idx'},inplace=True)
                df_edge = pd.concat([df_edge,df_edge2],axis=0)
                table.df_edge = df_edge
            
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def degree_bins(self,table):
        old_columns = table.df.columns
        df = table.df
        
#        if not table.undirected_graph:
#            bins = [0,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,5000,df['degree'].max()]#d
#        elif table.ori_columns.shape[0]==0:
#            bins = [0,1,2,3,4,df['degree'].max()]#e
#        elif table.ori_columns.shape[0]>1000:
#            bins = [-1,0,1,2,3,4,5,6,7,8,9,10,15,20,50,df['degree'].max()]
#        else:
#            bins = [-1,0,1,2,3,4,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,df['degree'].max()]
#        
#        bins.sort()
#        LOGGER.info(f'degree bins, {bins}')
#        df['degree_bins'] = pd.Categorical(pd.cut(df['degree'],bins)).codes


#        bins = int(max(10,df['degree'].nunique()/10))
#        threshold = df['degree'].quantile(1-1/bins)
#        
#        df['degree_bins'] = bins
#        df.loc[df['degree']<=threshold,'degree_bins'] = pd.Categorical(pd.cut(df.loc[df['degree']<=threshold,'degree'],bins)).codes
        
        
#        bins = 100
#        df['degree_bins'] = pd.qcut(df['degree'],bins,duplicates='drop',labels=False)
        
        
        bins = int(max(30,df['degree'].nunique()/10))
        
        df_tmp = df['degree'].value_counts().reset_index()
        df_tmp = df_tmp.rename(columns={'index':'degree','degree':'nums'})
        df_tmp = df_tmp.sort_values('degree')
        
        min_nums = df.shape[0]/bins
        k = 0
        cum_nums = 0
        bins_dict = {}
        for i,j in zip(df_tmp['degree'],df_tmp['nums']):
            cum_nums += j
            bins_dict[i] = k
            if cum_nums>=min_nums:
                k += 1
                cum_nums = 0
            
        df['degree_bins'] = df['degree'].map(bins_dict)
        print(bins_dict)
        
        print(df['degree'].value_counts())
        print(df['degree_bins'].value_counts())
        
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def bin_2_neighbor_mean_degree_bins(self,table):
        old_columns = table.df.columns
        df = table.df
        
        if not table.undirected_graph:
            df['2-neighbor_mean_degree_bins'] = df['2_out-neighbor_mean_degree_bins']+df['2_in-neighbor_mean_degree_bins']
        
#        bins = int(min(100,(df['2-neighbor_mean_degree_bins']/0.1).astype(int).nunique()))
#        threshold = df['2-neighbor_mean_degree_bins'].quantile(1-1/bins)
#        
#        df['bin_2-neighbor_mean_degree_bins'] = bins
#        df.loc[df['2-neighbor_mean_degree_bins']<=threshold,'bin_2-neighbor_mean_degree_bins'] = pd.Categorical(pd.cut(df.loc[df['2-neighbor_mean_degree_bins']<=threshold,'2-neighbor_mean_degree_bins'],bins)).codes

        
        
        
        bins = int(min(100,(df['2-neighbor_mean_degree_bins']/0.1).astype(int).nunique()))
        
        df_tmp = df['2-neighbor_mean_degree_bins'].value_counts().reset_index()
        df_tmp = df_tmp.rename(columns={'index':'degree','2-neighbor_mean_degree_bins':'nums'})
        df_tmp = df_tmp.sort_values('degree')
        
        min_nums = df.shape[0]/bins
        k = 0
        cum_nums = 0
        bins_dict = {}
        for i,j in zip(df_tmp['degree'],df_tmp['nums']):
            cum_nums += j
            bins_dict[i] = k
            if cum_nums>=min_nums:
                k += 1
                cum_nums = 0
            
        df['bin_2-neighbor_mean_degree_bins'] = df['2-neighbor_mean_degree_bins'].map(bins_dict)
        
        
        print(df['bin_2-neighbor_mean_degree_bins'].value_counts())
        
        if not table.undirected_graph:
            df = df.drop(columns='2-neighbor_mean_degree_bins')
        
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def edge_weight(self,table):
        old_columns = table.df.columns
        df = table.df
        df_edge = table.df_edge
        
        if df_edge['edge_weight'].nunique()!=1:
            if table.undirected_graph:
                df['weight_sum'] = df['node_index'].map(df_edge.groupby('src_idx')['edge_weight'].sum().to_dict())
                df['weight_sum'].fillna(0,inplace=True)
                df['weight_mean'] = df['weight_sum']*(1/(df['degree']+1e-9).to_numpy())
                df['weight_mean'].fillna(0,inplace=True)
            else:
                df['out_weight_sum'] = df['node_index'].map(df_edge.groupby('src_idx')['edge_weight'].sum().to_dict())
                df['in_weight_sum'] = df['node_index'].map(df_edge.groupby('dst_idx')['edge_weight'].sum().to_dict())
                df['out_weight_sum'].fillna(0,inplace=True)
                df['in_weight_sum'].fillna(0,inplace=True)
                
                df['weight_sum'] = df['out_weight_sum']+df['in_weight_sum']
                df['weight_sum'].fillna(0,inplace=True)
                
                df['out_weight_mean'] = df['out_weight_sum']*(1/(df['out_degree']+1e-9).to_numpy())
                df['in_weight_mean'] = df['in_weight_sum']*(1/(df['in_degree']+1e-9).to_numpy())
                df['weight_mean'] = df['weight_sum']*(1/(df['degree']+1e-9).to_numpy())
                
                df['out_weight_mean'].fillna(0,inplace=True)
                df['in_weight_mean'].fillna(0,inplace=True)  
                df['weight_mean'].fillna(0,inplace=True)  
        table.df = df
        return df.columns.drop(old_columns)
        
    @timeclass('Feat')
    def edge_weight_bins(self,table):
        old_columns = table.df.columns
        df = table.df
        df_edge = table.df_edge.copy()
        
        if df_edge['edge_weight'].nunique()!=1:
            if table.undirected_graph:
                pass
            else:
                bins = [-1,0,1,2,3,4,5,6,10,df_edge['edge_weight'].max()]#d
                bins.sort()
                
                LOGGER.info(f'edge weight bins, {bins}')
                df_edge['edge_weight_bins'] = pd.Categorical(pd.cut(df_edge['edge_weight'],bins)).codes
                
                df_out = df_edge.groupby('src_idx')['edge_weight_bins'].value_counts()
                df_out.name = 'num'
                df_out = df_out.reset_index()
                df_out.rename(columns={'src_idx':'node_index'},inplace=True)
                df_out = pd.pivot_table(df_out,index='node_index',columns='edge_weight_bins',values='num')
                df_out = df_out.fillna(0)
                
                #求比例
#                df_out = df_out.divide(df_out.values.sum(axis=1),axis=0)
                
                df_out.columns = [f'{df_out.columns.name}_{i}' for i in df_out.columns]
                
                df = df.merge(df_out,how='left',on='node_index')
                df[df_out.columns] = df[df_out.columns].fillna(0)
        table.df = df
        return df.columns.drop(old_columns)
            
    
    @timeclass('Feat')
    def neighbor(self,edge_matrix,df_value,columns_name,mean=None,namepsace=None):
        df_neighbor = pd.DataFrame(sparse_dot(edge_matrix,df_value),columns=[f'{namepsace}-neighbor_sum_{col}' for col in columns_name])
        if mean is not None:
            df_neighbor = pd.DataFrame(df_neighbor.to_numpy()*(1/(mean.to_numpy()+1)).reshape(-1,1),columns=df_neighbor.columns)
            df_neighbor.rename(columns={f'{namepsace}-neighbor_sum_{col}':f'{namepsace}-neighbor_mean_{col}' for col in columns_name},inplace=True)
            
        return df_neighbor
    
    @timeclass('Feat')
    def get_neighbor(self,table):
        old_columns = table.df.columns
        columns_name = [i for i in table.df.columns if i not in ['node_index','is_test','label']]
        edge_matrix = table.edge_matrix
        df = table.df
        df_value = df[columns_name]
        if table.undirected_graph:
            df_neighbor = self.neighbor(edge_matrix,df_value.values,columns_name,mean=df['degree'],namepsace=1)
            df_neighbor2 = self.neighbor(edge_matrix,df_neighbor.values,columns_name,mean=df['degree'],namepsace=2)
            
            if table.unbalanced:
                df_neighbor3 = self.neighbor(edge_matrix,df_neighbor2.values,columns_name,mean=df['degree'],namepsace=3)
                df = pd.concat([df,df_neighbor,df_neighbor2,df_neighbor3],axis=1)
            else:
                df = pd.concat([df,df_neighbor2],axis=1)
        else:
            out_df_neighbor = self.neighbor(edge_matrix,df_value.values,columns_name,mean=df['out_degree'],namepsace='1_out')
            out_df_neighbor2 = self.neighbor(edge_matrix,out_df_neighbor.values,columns_name,mean=df['out_degree'],namepsace='2_out')
#            out_df_neighbor3 = self.neighbor(edge_matrix,out_df_neighbor3.values,columns_name,mean=df['out_degree'],namepsace='3_out')
            
            in_df_neighbor = self.neighbor(edge_matrix.T,df_value.values,columns_name,mean=df['in_degree'],namepsace='1_in')
            in_df_neighbor2 = self.neighbor(edge_matrix.T,in_df_neighbor.values,columns_name,mean=df['in_degree'],namepsace='2_in')
#            in_df_neighbor3 = self.neighbor(edge_matrix.T,in_df_neighbor2.values,columns_name,mean=df['in_degree'],namepsace='3_in')
            
            df = pd.concat([df,out_df_neighbor2,in_df_neighbor2],axis=1)
        
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def reverse_decode(self,table):
        ori_columns = table.ori_columns
        df = table.df
        if ori_columns is not None and (df[ori_columns]>1).sum().sum()==0:
            size = 10
            columns_code = np.array([2**i for i in range(size)]).reshape(-1,1)
            columns_n = ori_columns.shape[0]
            nums = columns_n//size
            df_tmp = df[ori_columns].values
            
            for i in range(nums):
                #reverse_one_hot_f->rf
                df[f'rf{i}'] = np.dot(df_tmp[:,i*size:(i+1)*size],columns_code)
 
            if columns_n%size!=0:
                nums += 1
                i += 1
                df[f'rf{i}'] = np.dot(df_tmp[:,i*size:columns_n],np.array([2**i for i in range(columns_n%size)]).reshape(-1,1))
 
    #            table.categories.extend([f'rf{i}' for i in range(nums)])
        table.df = df
    
    @timeclass('Feat')
    def feat_sum(self,table):
        table.df['feature_sum'] = table.df[table.ori_columns].values.sum(axis=1)
        
    @timeclass('Feat')
    def neighbor_label(self,table):
        old_columns = table.df.columns
        #如果是有向图，则此只写了out邻居
        df = table.df
        df_edge = table.df_edge.copy()
        df_edge = df_edge.rename(columns={'src_idx':'node_index'})
        label_dic = df[['node_index','label']].set_index('node_index').to_dict()['label']
        df_edge['around_label'] = df_edge['dst_idx'].map(label_dic)
        
        df_tmp = df_edge.groupby('node_index')['around_label'].value_counts()
        df_tmp.name = 'num'
        df_tmp = df_tmp.to_frame()
        
#        df_tmp = df_tmp.reset_index()
#        df_tmp['degree'] = df_tmp['node_index'].map(df[['node_index','degree']].set_index('node_index').to_dict()['degree'])
#        df_tmp['num'] = df_tmp['num']/df_tmp['degree']
        
        df_tmp = df_tmp.pivot_table(index='node_index',columns='around_label',values='num')
        
        df_tmp.columns = [f'{df_tmp.columns.name}_{i}' for i in df_tmp.columns]
        df = df.merge(df_tmp,how='left',on='node_index')
        table.df = df
        
        return df.columns.drop(old_columns)
        
    @timeclass('Feat')
    def most_neighbor_label(self,table):
        #如果是有向图，则此只写了out邻居
        df = table.df
        df_edge = table.df_edge.copy()
        df_edge = df_edge.rename(columns={'src_idx':'node_index'})
        label_dic = df[['node_index','label']].set_index('node_index').to_dict()['label']
        df_edge['around_label'] = df_edge['dst_idx'].map(label_dic)
        
        df_tmp = df_edge.groupby('node_index')['around_label'].value_counts()
        df_tmp.name = 'num'
        df_tmp = df_tmp.reset_index()
        
        df_tmp = df_tmp.iloc[df_tmp.groupby('node_index')['num'].idxmax()]
        df['most_neighbor_label'] = df['node_index'].map(df_tmp.set_index('node_index').to_dict()['around_label'])
#        df['most_neighbor_label'] = df['most_neighbor_label'].fillna(df['label'].mode()[0])
        
        table.df = df
        
    @timeclass('Feat')
    def reverse_decode_value_counts(self,table):
        ori_columns = table.ori_columns
        df = table.df
        if ori_columns is not None and (df[ori_columns]>1).sum().sum()==0:
            size = 100
            columns_n = ori_columns.shape[0]
            nums = int(np.ceil(columns_n/size))
            for i in range(nums):
                #reverse_one_hot_f_value_count->rf_vc
                df[f'rf_vc{i}'] = np.argmax(df[[f'f{j}' for j in range(i*size,min((i+1)*size,columns_n))]].values,axis=1)
                df[f'rf_vc{i}'] = df[f'rf_vc{i}'].map(df[f'rf_vc{i}'].value_counts())
        table.df = df
        
    @timeclass('Feat')
    def spectral_clustering(self,table):
        if table.undirected_graph:
            k = 100
            df = table.df
            W = table.edge_matrix.copy()
            W[np.diag_indices_from(W)] = 0
            D = np.diag(df['degree'])
            L = D-W
            D_half = np.diag(df['degree']**(-1/2))
            Standard_L = sparse_dot(sparse_dot(D_half,L),D_half)
            Standard_L.fill(0)#这里会出现空值，一脸懵逼啊
            eigenvalue,featurevector = np.linalg.eig(Standard_L)
            spectral_F = featurevector[:,-k:]
            spectral_F = spectral_F/spectral_F.sum(axis=0)
            spectral_F = np.real(spectral_F)
            df = pd.concat([df,pd.DataFrame(spectral_F,columns=[f'spectral_F{i}' for i in range(k)])],axis=1)
            table.df = df
    
    @timeclass('Feat')
    def graph_embedding(self,table):
        old_columns = table.df.columns
        
        import torch
        import torch.optim as optim
        from torch_geometric.nn import Node2Vec
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        df = table.df
        n = df.shape[0]
        
        embedding_dim = 10#n//200
        walk_length = 4#n//100
        context_size = 4#min(30,walk_length//3)
        node2vec = Node2Vec(num_nodes=n,embedding_dim=embedding_dim,
                            walk_length=walk_length,context_size=context_size,
                 walks_per_node=1,p=1,q=1,num_negative_samples=None).to(device)
        
        edge_index = torch.tensor(table.df_edge[['src_idx','dst_idx']].values).T
        
        node2vec.to(device)
        subset = torch.tensor(df['node_index']).to(device)
        edge_index = edge_index.to(device)
        
        optimizer = optim.Adam(node2vec.parameters(), lr=0.05)
        
        node2vec.train()
        
        best_loss = np.inf
        keep_epoch = 0
        for i in range(500):
            optimizer.zero_grad()
            loss = node2vec.loss(edge_index, subset=subset)
            loss.backward()
            optimizer.step()
            print(f'epoch:{i} loss:{loss}')
            if best_loss>loss:
                best_loss = loss
                keep_epoch = 0
            else:
                keep_epoch += 1
                if keep_epoch > 10:
                    break
        
        df = pd.concat([df,pd.DataFrame(node2vec(subset).detach().cpu().numpy(),columns=[f'node_embedding_{i}' for i in range(embedding_dim)])],axis=1)
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def connected_subgraph(self,table):
        old_columns = table.df.columns
        
        df = table.df
        g = nx.Graph()
        #nx.DiGraph
        
        g.add_nodes_from(df['node_index'])
        g.add_edges_from(table.df_edge[['src_idx', 'dst_idx']].values)
#        g.add_weighted_edges_from(table.df_edge[['src_idx', 'dst_idx','edge_weight']].values)
        
        df['subgraph'] = 0
        i = 0
        for subgraph in nx.connected_components(g):
            df.loc[list(subgraph),'subgraph'] = i
            i += 1
        
        #de加上就崩 看看为啥
        LOGGER.info(f"subgraph \n{df['subgraph'].value_counts()}")
        
#        df['subgraph_node_num'] = df['subgraph'].map(df['subgraph'].value_counts())
#        
#        df['subgraph_node_num'] = pd.Categorical(df['subgraph_node_num']).codes
        
#        df = df.drop(columns='subgraph')
        
        
#        bins = [-1,0,1,2,3,4,5,10,df['subgraph_node_num'].max()]
#        bins.sort()
#        
#        df['subgraph_node_num'] = pd.Categorical(pd.cut(df['subgraph_node_num'],bins)).codes
#        df['subgraph_node_num'] = pd.Categorical(df['subgraph_node_num']).codes
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def fr(self,table):
        def sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-5, dim=3):
            # np.random.seed(1)
            nodes_num = A.shape[0]
            A = A.tolil()
            A = A.astype('float')
            if pos is None:
                pos = np.asarray(np.random.rand(nodes_num, dim), dtype=A.dtype)
                print('Init pos', pos)
            else:
                pos = np.array(pos)
                pos = pos.astype(A.dtype)
        
            if fixed is None:
                fixed = []
        
            if k is None:
                k = np.sqrt(1.0 / nodes_num)
            t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
            dt = t / float(iterations + 1)
            displacement = np.zeros((dim, nodes_num))
            for iteration in range(iterations):
                displacement *= 0
                for i in range(A.shape[0]):
                    if i in fixed:
                        continue
                    delta = (pos[i] - pos).T
                    distance = np.sqrt((delta ** 2).sum(axis=0))
                    distance = np.where(distance < 0.01, 0.01, distance)
                    Ai = np.asarray(A.getrowview(i).toarray())
#                    print('Ai', Ai)
                    displacement[:, i] += \
                        (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
                # update positions
                length = np.sqrt((displacement ** 2).sum(axis=0))
                length = np.where(length < 0.01, 0.1, length)
                delta_pos = (displacement * t / length).T
                pos += delta_pos
                # cool temperature
                t -= dt
                err = np.linalg.norm(delta_pos) / nodes_num
                if err < threshold:
                    break
        
            return pos
        import numpy as np
        from scipy.sparse import coo_matrix
        n = table.df.shape[0]
        df_edge = table.df_edge
        A = coo_matrix((df_edge['edge_weight'],(df_edge[['src_idx', 'dst_idx']].values.T)),shape=(n,n))
        pos = sparse_fruchterman_reingold(A)
        pos = pd.DataFrame(pos,columns=[f'fr_{i}' for i in range(pos.shape[1])])
        df = table.df
        old_columns = table.df.columns
        df = pd.concat([df,pos],axis=1)
        table.df = df
        return df.columns.drop(old_columns)

        
#    @timeclass('Feat')
#    def graph_embedding2(self,table):
#        x = torch.tensor(table.df['node_index'].values)
#        edge_index = table.df_edge[['src_idx', 'dst_idx']].to_numpy()
#        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
#        edge_weight = table.df_edge['edge_weight'].to_numpy()
#        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
#        
#        gcn_model = GCN_Embedding(node_nums=x.shape[0],dim=128)
#        with torch.no_grad():
#            res = gcn_model(x,edge_index,edge_weight)
#            res = pd.DataFrame(res.numpy(),columns=[f'gcn_embeddings{i}' for i in range(res.shape[1])])
#        
#        table.df = pd.concat([table.df,res],axis=1)
#        
#        
#import torch
#from torch_geometric.nn import GCNConv
#import torch.nn.functional as F
#
#class GCN_Embedding(torch.nn.Module):
#    def __init__(self, node_nums=1000, dim=16):
#        super(GCN_Embedding, self).__init__()
#        self.node_embeddings = torch.nn.Embedding(node_nums, dim)
#
#        self.conv1 = GCNConv(dim, 512)
#        self.conv2 = GCNConv(512, 128)
#        self.conv3 = GCNConv(128, 32)
#
#    def forward(self,x,edge_index,edge_weight):
#        x = self.node_embeddings(x)
#        x = F.relu(x)
#        x = self.conv1(x,edge_index,edge_weight)
#        x = F.dropout(x,p=0.5)
#        x = F.relu(x)
#        x = self.conv2(x,edge_index,edge_weight)
#        x = F.dropout(x,p=0.5)
#        x = F.relu(x)
#        x = self.conv3(x,edge_index,edge_weight)
#        return x

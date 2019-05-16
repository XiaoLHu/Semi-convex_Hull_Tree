/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms of license.
 *
 */
#include<stdio.h>
#ifndef CONVEX_TREE_H_INCLUDED
#define CONVEX_TREE_H_INCLUDED
#include <vector>
#include <time.h>
#include <winable.h>
#include "ConvexNode.h"
#include "basic_functions.h"
#include "openCL_tool.h"
#include "cyw_timer.h"
#include "cyw_types.h"

//the following const is used to specify the detail algorithm used in kNN_cpu
//0-use approximate qp and recursive (kNN_by_recusive) algorithm
#define USE_CPU_RECURSIVE_APPRXIMATE_QP 0
//1-use exact qp and recursive algorithm
#define USE_CPU_RECURSIVE_QP 1
//2-use approximate qp and leaf algorithm (kNN_by_leaf) algorithm
#define USE_CPU_LEAF_APPRXIMATE_QP 2
//3-use Exact qp and leaf algorithm (kNN_by_leaf) algorithm
#define USE_CPU_LEAF_QP 3
//4- GPU version that use openCL and approximate QP to implement kNN
#define USE_GPU_RECURSIVE_APPRXIMATE_QP 4
//4- GPU version that use openCL and leaf QP to implement kNN
#define USE_GPU_LEAF_APPRXIMATE_QP 5
#define USE_GPU_CIRCLE_QUEUE_APPRXIMATE_QP 6

int GPU_PLATFORM_ID=1;

//int GPU_PLATFORM_ID=0;

// binary convex tree
/*
   dist        -- the NN dist from query point
   dist_square -- the square of dist
   dist_shift  -- the quadprog return the min dist which without adding the item "q'*q" where q is the query point
*/
template<class T>
class NNResult{
public:
    static Vector<NNResult<T>*> create_an_NNResult_vec(int num, int K){
        Vector<NNResult<T>*> result;
        NNResult<T> *NN_result_arr=new NNResult<T> [num];
        FLOAT_TYPE  *k_dists_squre_arr= new FLOAT_TYPE[num*K];
        FLOAT_TYPE  *k_dists_arr= new FLOAT_TYPE[num*K];
        int         *k_indexes_arr=new int[num*K];

        //--------------------------init-------------------------
        for (int i=0;i<num*K;i++){
            k_dists_squre_arr[i]=INFINITE;
            k_dists_arr[i]=INFINITE;
        }
        memset(k_indexes_arr,-1,num*K);
        //--------------------------init-------------------------

        result.resize(num);
        for (int i=0;i<num;i++){
            NN_result_arr[i].k_dists_squre=k_dists_squre_arr+i*K;
            NN_result_arr[i].k_dists=k_dists_arr+i*K;
            NN_result_arr[i].k_indexes=k_indexes_arr+i*K;
            NN_result_arr[i].K=K;
            result[i]= &NN_result_arr[i];
        }
        return result;
    }

    T cur_kth_dist_shift=INFINITE;     //the k^{th} distance shift
    int index=-1;
    int K=1;
    FLOAT_TYPE* k_dists_squre=NULL;    //all k NN dists_square
    FLOAT_TYPE* k_dists=NULL;          //all k NN dists
    int* k_indexes=NULL;               //all k NN indexes

    //initialize
    void init(int k){
        K=k;
        free_stru();
        k_dists_squre= new FLOAT_TYPE [K];
        k_dists= new FLOAT_TYPE [K];
        k_indexes= new int[K] ;
        for (int i=0;i<K;i++){
            k_dists_squre[i]= INFINITE;
            k_dists[i]= INFINITE;
            k_indexes[i]=-1;
        }
    }

    void free_stru(){
        if (k_dists_squre!=NULL){
            delete []k_dists_squre;
            delete []k_dists;
            delete []k_indexes ;
        }
    }

    T get_kth_dist() {
        return k_dists[K-1];
    }

    T get_kth_dist_square() {
        return k_dists_squre[K-1];
    }

    bool is_exists( int index){
        bool result=false;
        for (int i=0;i<K;i++){
            if (k_indexes[i]==index)
            return true;
        }
        return result;
    }
    //if the dist_square is less than the k_dists_squre[k-1],
    //then replace it with dist_square, and update
    bool update(FLOAT_TYPE dist_square, int index){
        bool result=false;
        if (k_dists_squre[K-1]>dist_square){
            k_dists_squre[K-1]=dist_square;
            k_dists[K-1]=sqrt(dist_square);
            k_indexes[K-1]=index;

            //swap the dists square and dists in order to order the dists and dist_square
            //as well as the indexes
            int j=K-1;
            for(;j>0;j--){
                if(k_dists_squre[j-1] > k_dists_squre[j]){
                    //swap distances and square
                    FLOAT_TYPE tmp=k_dists_squre[j-1];
                    k_dists_squre[j-1]=k_dists_squre[j];
                    k_dists_squre[j]=tmp;

                    tmp=k_dists[j-1];
                    k_dists[j-1]=k_dists[j];
                    k_dists[j]=tmp;

                    //swap indices
                    int tmp_idx=k_indexes[j-1];
                    k_indexes[j-1]=k_indexes[j];
                    k_indexes[j]=tmp_idx;
                } // end if
            }//end for
            result=true;
        }
        return result;
    }


    void print(){
        //std::cout <<"dist=" << dist<<", shift dist=" << dist_shift<<", index="<< index<<"\n";
        std::cout <<"\n     the K neighbor indexes are:";
        for (int i=0; i<K;i++){
            std::cout << k_indexes[i]<<", ";
        }
        std::cout <<"\n     the K k_dists_squre are:";
        for (int i=0; i<K;i++){
            std::cout << k_dists_squre[i]<<", ";
        }
    }
};

//this is an abstract class
template<typename T>
class Convex_Tree
{
public:
    //This is the entry for user to perform task of finding kNN
    void kNN(Matrix<T> &query_points_mat, int K);

    void brute_force_kNN(Matrix<T> &query_points_mat, int K);

    void set_alg_type(int alg_type){
        this->ALG_TYPE=alg_type;
    }

    Convex_Tree();

    //leaf_pts_percent : the percent that specify the number of leaf_pts_num
    Convex_Tree(Matrix<T>& data, FLOAT_TYPE leaf_pts_percent, int alg_type);

    Convex_Tree(Matrix<T>& data, int leaf_pts_num, int alg_type);

    virtual ~Convex_Tree(){
        //free nodes
        for (int i=0;i<this->nodes.size();i++)
            delete this->nodes[i];
    };

    Matrix<T>  &getData(){ return m_data;};
    Matrix<T>  &getSortedData(){ return this->m_sorted_data;};
    Vector<int>&getSortedData_ori_indexes(){ return this->m_sorted_data_ori_indexes;};
    Vector<T>  &getDists_Square(){return this->m_dists_square_v;};

    inline void clearNN_info();

    virtual void print_kNN_running_time_info();
    virtual void save_KNN_running_time_info(FILE* log_file);

    void print_tree_info();
    void save_tree_info(FILE* log_file);


    void print_kNN_reult(){
        //this is virtual procedure
        this->do_print_kNN_result();
    }

    //record and count the number of active constrains for query point in all nodes
    Vector<int> count_active_constrains_num_for_a_query_point(Vector<FLOAT_TYPE>& query_point);
    //trace a node to root
    std::vector<int> trace_a_node_to_root(int node_index, int print=1);

    //check whether all points of a node are inside the node
    bool check_node_valid(int check_node_index);

    bool check_all_leaf_nodes_valid();

    //check whether all points of a node are inside the node
    bool check_sub_nodes_valid(int check_node_index, int node_index);

    //find the nearest neighbor in nodes[node_index]
    FLOAT_TYPE find_nn_in_node(int node_index, Vector<T>& q);

    //after kNN processing, get the result by the following three entries
    //they are all virtual procedures, should be override in subclass.
    virtual FLOAT_TYPE*  get_kNN_dists(int query_point_index){
    };

    virtual FLOAT_TYPE*  get_kNN_dists_squre(int query_point_index){
    };

    virtual int* get_kNN_indexes(int query_point_index){
    };

    //GPU_Convex_tree will override it
    virtual void set_batch_size(int batch_size){
        //do nothing
        return;
    }

    //find the node where a point locates.
    int find_node_index_for_a_point(int point_index);

protected:
    //this should be implemented in subclass
    void virtual do_kNN(Matrix<T> &query_points_mat)
    {
        //do nothing here, need to be implemented in subclass
    }

    //this should be implemented in subclass
    void virtual do_brute_force_kNN(Matrix<T> &query_points_mat)
    {
        //do nothing here, need to be implemented in subclass
    }

    /*
        this is used for subclass to init specific task, it will be call in init_process_query_points
        defined in Convex_tree, and should be override in subclass.
        e.g., in GPU_Convex_tree override it and perform init_openCL_device and init_opecl_parameters
    */
    void virtual init_kNN_result(){
        //do nothing here, need to be implemented in subclass
    };

    void virtual do_print_kNN_result(){
    }

    std::vector<ConvexNode<T>*> nodes;

    //this is a simplified tree structure of all convex_nodes, and used as a parameter that
    //pass to openCL device. It will be initialized after build_tree.
    Vector<CONVEX_TREE> simplified_tree;

    //this is used to specify which searching algorithm used in kNN_CPU
    int ALG_TYPE=USE_CPU_RECURSIVE_APPRXIMATE_QP;

    //set default K=1;
    int  K=1;
    int  depth=1;
    int  nodes_num=0;
    int  leaf_nodes_num=0;
    long quadprog_cal_num=0;
    int  quadprog_cal_time=0;

    //saves the number of scalar product in 'is_appr_min_dist_from_q_larger_by_hyper_plane'
    //one scalar product is equal to a dist computation.
    long scalar_product_num=0;
    float leaf_pts_percent=0.0;
    int  leaf_pts_num_threshold=0;
    long long dist_computation_times_in_host=0;
    //---------------------timers------------------------------
    CYW_TIMER timer_whole_alg;
    CYW_TIMER timer_build_tree;
    CYW_TIMER timer_total_quadprog;
    //record the running time to update kNN dists and indexes
    CYW_TIMER timer_total_update_kNN_dists;
    CYW_TIMER timer_total_dists_computation;
    CYW_TIMER timer_init_process_query_points;
    //record the running time to search leaf node for a query point
    CYW_TIMER timer_total_approximate_searching;
    //---------------------timers------------------------------

    //the query points saved in Vector
    Vector<Vector<T> > query_points_vec;

    //for i^th query point, appr_leaf_node_indexes[i] saves the leaf node index handled in approximateNN
    Vector<int> appr_leaf_node_indexes;

    //approximate NNResult
    //NNResult<T> app_NNResult;
    //raw data
    Matrix<T> m_data;

    //after building bkm tree, the data will be rearranged according
    //to the order of leaf nodes
    Matrix<T> m_sorted_data;
    Vector<int> m_sorted_data_ori_indexes;
    //save the offsets of the data save in each leaf node in m_sorted_data
    std::vector<int> leaf_nodes_start_pos_in_data_set;


    bool ALIGN_SORTED_DATA=false;

    //after building bkm tree, the data will be rearranged according
    //to the order of leaf nodes, and the size of all data saved in each leaf node will
    //be aligned to multiple DATA_ALIGNED_SIZE bit. If it is not, then some null space will be filled
    //into m_sorted_aligned_data, which makes the multiple DATA_ALIGNED_SIZE bit hold.
    //It is useful for many-cores device to compute.
    int DATA_ALIGNED_SIZE=32;
    Matrix<T> m_aligned_sorted_data;
    Vector<int> m_aligned_sorted_data_ori_indexes;
    //save the offsets of the aligned data save in each leaf node in m_aligned_sorted_data
    std::vector<int> aligned_leaf_nodes_start_pos_in_data_set;

    //this is used to store all distance square from query point to all point in m_sorted_data,
    //the distances square of those filtered point are not computed.
    Vector<T> m_dists_square_v;

    //the original index of each leaf node over all nodes which is different from the
    //leave index save in each leaf node. a leaf node's leave is the index of the
    //leave node over all leaf nodes
    std::vector<int> leaf_nodes_ori_indexes;

    int WORKGROUP_SIZE_BRUTE=32;
    //int BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE=1000000;
    int BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE=20000;
    //int BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE=10000;
private:
    //curent approximate NNResult. in each iteration it save the current kNN result.

    /*------------------------------init query points and K---------------------------*/
    void init_process_query_points(Matrix<T> &query_points_mat, int K);
    void init_query_points_info   (Matrix<T>& query_points_mat);
    /*------------------------------init query points and K---------------------------*/

    inline FLOAT_TYPE getBeta  (Vector<FLOAT_TYPE>& alpha, Vector<T> &point, Vector<int>& indexes);
    inline FLOAT_TYPE getBeta_1(Vector<FLOAT_TYPE>& alpha, FLOAT_TYPE beta, Vector<int>& indexes);
    //sort m_data after bkm tree is built by the order of leave nodes
    void sort_raw_data();

    void align_sorted_raw_data(int aligned_len);

    int  build_tree(int cur_depth, Vector<T>&alpha,  T beta,
                    int parent_node_index, Vector<int>& index_vec,
                    Vector<T> cur_center);

    /*
        build simplified_tree which is a simplified tree structure of all convex_nodes, and
        used as a parameter that can be passed to openCL device. 'build_simplified_tree' will
        be called after build_tree.
    */
    void build_simplified_tree();

};


//record and count the number of active constrains for query point in all nodes
template <typename T>
Vector<int> Convex_Tree<T>::count_active_constrains_num_for_a_query_point(Vector<FLOAT_TYPE>& query_point){
    //current the number of constrains on each node are less than 20.
    Vector<int> result(20);
    result=0;
    for (int i=0;i<this->nodes.size();i++){
        if (i==0) continue;
        int tmp_num=this->nodes[i]->count_active_constrains(query_point);
        if (result.size() > tmp_num){
            result[tmp_num]++;
        }
    }
    return result;
}


//trace a node to root
template <typename T>
std::vector<int> Convex_Tree<T>::trace_a_node_to_root(int node_index, int print){
    std::vector<int> result;
    ConvexNode<T>* tmp_node= this->nodes[node_index];
    if (print)
        std::cout<<"The order to root: "<<tmp_node->index<<", ";
    result.push_back(node_index);
    int parent_node_index=tmp_node->parent_index;
    while (parent_node_index!=-1){
        result.push_back(parent_node_index);
        tmp_node= this->nodes[parent_node_index];
        if (print)
            std::cout<<tmp_node->index<<", ";
        parent_node_index=tmp_node->parent_index;
    }
    return result;
}

template <typename T>
Convex_Tree<T>::Convex_Tree():m_data(NULL),leaf_pts_num_threshold(0)
{
}


//directly search leaves not from top to bottom
template <typename T>
inline void Convex_Tree<T>::clearNN_info()
{
    //clear before running time
    this->timer_init_process_query_points.init_my_timer();
    this->timer_total_approximate_searching.init_my_timer();
    this->timer_total_dists_computation.init_my_timer();
    this->timer_total_quadprog.init_my_timer();
    this->timer_total_update_kNN_dists.init_my_timer();

    //clear before statistic info
    this->quadprog_cal_num=0;
    this->quadprog_cal_time=0;
    this->dist_computation_times_in_host=0;
}


template <typename T>
void Convex_Tree<T>::init_process_query_points(Matrix<T> &query_points_mat, int K){
    this->timer_init_process_query_points.start_my_timer();
    this->K=K;
    this->init_query_points_info(query_points_mat);

    //this is virtual procedure for subclass to override it.
    this->init_kNN_result();
    this->timer_init_process_query_points.stop_my_timer();
}

//the entry for perform kNN
template <typename T>
void Convex_Tree<T>:: kNN(Matrix<T> &query_points_mat, int K){
    int dim = query_points_mat.ncols();
    if (dim!= this->m_data.ncols()){
        std::cout<<"the dim of query points is "<< dim<<", while the dim of data set is "<<m_data.ncols();
        return;
    }

    this->timer_whole_alg.start_my_timer();
    this->clearNN_info();
    this->init_process_query_points(query_points_mat,K);
    //virtual procedure that need to be override in subclass
    this->do_kNN(query_points_mat);
    this->timer_whole_alg.stop_my_timer();
}

//the entry for perform kNN
template <typename T>
void Convex_Tree<T>:: brute_force_kNN(Matrix<T> &query_points_mat, int K){
    int dim = query_points_mat.ncols();
    if (dim!= this->m_data.ncols()){
        std::cout<<"the dim of query points is "<< dim<<", while the dim of data set is "<<m_data.ncols();
        return;
    }

    this->timer_whole_alg.start_my_timer();
    this->clearNN_info();
    this->init_process_query_points(query_points_mat,K);
    //virtual procedure that need to be override in subclass
    this->do_brute_force_kNN(query_points_mat);
    this->timer_whole_alg.stop_my_timer();
}

//ctor
template<typename T>
Convex_Tree<T>::Convex_Tree(Matrix<T>& data,
                            int leaf_pts_num,
                            int alg_type):m_data(data),leaf_pts_num_threshold(leaf_pts_num),ALG_TYPE(alg_type)
{
    std::cout<<"\nbuilding tree...\n";
    Vector<FLOAT_TYPE> alpha(1);
    Vector<int> index_vec(1);
    Vector<FLOAT_TYPE> cur_center(0);
    build_tree(this->depth,alpha,0,-1,index_vec,cur_center);

    std::cout<<"convex tree building finished.";
    //save indexes of leave nodes
    for (int i=0; i<this->nodes_num;i++){
        if (this->nodes[i].isLeaf){
            this->leave_nodes.push_back(i);
        }
    }

    std::cout<<"OpenCL devices is initialized.";

    //rearrange the raw data, result in sorted_data
    sort_raw_data();

    m_dists_square_v= m_dists_square_v.resize(data.nrows());
}

//leaf_pts_percent is a percent that specify the leaf_num= (data.nrows() * leaf_pts_percent)
template<typename T>
Convex_Tree<T>::Convex_Tree(Matrix<T>& data,
                            FLOAT_TYPE leaf_pts_percent,
                            int alg_type):m_data(data),leaf_pts_percent(leaf_pts_percent),ALG_TYPE(alg_type)
{
    if ((leaf_pts_percent>=0.5)||(leaf_pts_percent<=0))
            throw std::runtime_error("illegal percent, it should be in (0,1)!");

    this->leaf_pts_num_threshold= (int) (data.nrows() * leaf_pts_percent);

    std::cout<<"\n building tree...\n";
    Vector<T> alpha(1);
    Vector<int> index_vec(1);
    Vector<T> cur_center(0);

    this->timer_build_tree.start_my_timer();
    //build tree
    build_tree(this->depth,alpha,0,-1,index_vec,cur_center);
    //build simplified_tree

    std::cout<<"building simplified tree...\n";
    this->build_simplified_tree();
    this->timer_build_tree.stop_my_timer();

    std::cout<<"sort raw data...\n";
    //rearrange the raw data, result in sorted_data
    sort_raw_data();

    //this is used to store all distance square from a query point to all point in m_sorted_data,
    //the distances square of those filtered point are not computed.
    this->m_dists_square_v.resize(m_data.nrows()) ;
};



template<typename T>
inline FLOAT_TYPE Convex_Tree<T>::getBeta(Vector<FLOAT_TYPE>& alpha, Vector<T> &point, Vector<int>& indexes){
    FLOAT_TYPE beta= -scalar_product(alpha, point);
    //shift the hyperplane
    beta=this->getBeta_1(alpha,beta,indexes);
    return beta;
}


//indexes saves the points' indexes that for all x\in indexes satisfied alpha'*x + beta>=0
template<typename T>
inline FLOAT_TYPE Convex_Tree<T>::getBeta_1(Vector<FLOAT_TYPE> &alpha, FLOAT_TYPE beta, Vector<int>& indexes){
    int rows=indexes.size();
    int cols= this->m_data.ncols();
    Vector <T> beta_v (rows);
    Vector<FLOAT_TYPE> dists(rows); //= alpha.product(data);

    for (int i = 0; i < rows; i++){
        FLOAT_TYPE tmp=0;
        for (int j=0;j<cols;j++){
            tmp+= this->m_data[indexes[i]][j]*alpha[j];
        }
        dists[i] =tmp ;
    }

    for (int i=0;i<rows;i++)
        dists[i]= dists[i] +beta;

    unsigned min_index= index_min(dists);
    int real_index= indexes[min_index];
    if (real_index> this->m_data.nrows()){
        std::cout<<"xxx";
    }

    //shift the hyperplane to newPoint
    FLOAT_TYPE newbeta=-scalar_product(alpha, this->m_data.extractRow(real_index));
    return newbeta;
}

//sort m_data after bkm tree is built.
template<typename T>
void Convex_Tree<T>::sort_raw_data(){
    register int i,j,  nodes_num=this->nodes.size();
    this->m_sorted_data.resize(m_data.nrows(),m_data.ncols());
    this->m_sorted_data_ori_indexes.resize(m_data.nrows());
    int cur_offset =0;
    int aligned_data_len=0;
    //int cur_leave_indx  =0;
    //std::set<unsigned int> indexes();
    Vector<int> indexes;
    for (i=0; i<nodes_num;i++)
    {
        if (this->nodes[i]->isLeaf)
        {
            //this->nodes[i]->leaf_index=cur_leave_indx;
            //the offset of the data stored in cur_node saved in sorted_data
            this->nodes[i]->saved_data_offset=cur_offset;
            this->leaf_nodes_ori_indexes.push_back(i);
            this->leaf_nodes_start_pos_in_data_set.push_back(cur_offset);
            int data_len=nodes[i]->data.nrows();
            float tmp=(float)data_len/this->DATA_ALIGNED_SIZE;
            aligned_data_len += ceil(tmp)*DATA_ALIGNED_SIZE;
            indexes.resize(data_len);
            for (j=0;j<data_len;j++){
                //indexes.insert(j+cur_offset);
                indexes[j]=j+cur_offset;
                m_sorted_data_ori_indexes[cur_offset+j]=nodes[i]->pts_indexes[j];
            }
            cur_offset+= data_len;
            m_sorted_data.setRows(indexes,nodes[i]->data);
            //indexes.clear();
            //cur_leave_indx++;
        }
    }


    //---align the sorted data such that all data saved in leaf node will be multiple 32-bit
    //---which is useful for processing in gpu devices.
    if (ALIGN_SORTED_DATA)
        align_sorted_raw_data(aligned_data_len);

}

//align the sorted data such as all data saved in leaf node will be multiple 32 bits
template<typename T>
void Convex_Tree<T>::align_sorted_raw_data(int aligned_len){
    register int i,j,  nodes_num=this->nodes.size();
    int cur_offset =0;
    this->m_aligned_sorted_data.resize(aligned_len,m_data.ncols());
    this->m_aligned_sorted_data_ori_indexes.resize(aligned_len);

    std::set<unsigned int> indexes;
    for (i=0; i<nodes_num;i++)
    {
        if (this->nodes[i]->isLeaf)
        {
            int data_len=nodes[i]->data.nrows();
            float tmp=(float)data_len/this->DATA_ALIGNED_SIZE;
            int aligned_data_len= ceil(tmp)*DATA_ALIGNED_SIZE;

            this->aligned_leaf_nodes_start_pos_in_data_set.push_back(cur_offset);

            for (j=0;j<data_len;j++){
                indexes.insert(j+cur_offset);
                m_aligned_sorted_data_ori_indexes[cur_offset+j]=nodes[i]->pts_indexes[j];
            }

            for (j=data_len;j<aligned_data_len;j++){
                m_aligned_sorted_data_ori_indexes[cur_offset+j]=-1;
            }

            this->m_aligned_sorted_data.setRows(indexes,nodes[i]->data);

            //this->m_aligned_sorted_data.print();
            //m_aligned_sorted_data.extractRows(indexes).print();

            indexes.clear();
            cur_offset+= aligned_data_len;
        }
    }
    //this->m_aligned_sorted_data.print();
    //m_aligned_sorted_data_ori_indexes.print("  ");

}



/*
    build simplified_tree which is a simplified tree structure of all convex_nodes, and
    used as a parameter that can be passed to openCL device. 'build_simplified_tree' will
    be called after build_tree.
*/
template<typename T>
void  Convex_Tree<T>::build_simplified_tree(){
    this->simplified_tree.resize(this->nodes_num);
    int leaf_index=0;
    for (int i=0; i<this->nodes_num;i++){
        this->simplified_tree[i].node_index     = this->nodes[i]->index;
        this->simplified_tree[i].isLeaf         = this->nodes[i]->isLeaf;
        this->simplified_tree[i].left_node      = this->nodes[i]->left;
        this->simplified_tree[i].right_node     = this->nodes[i]->right;
        this->simplified_tree[i].parent_index   = this->nodes[i]->parent_index;

        //if this node is not leaf node, leaf_index=-1, else leaf_index is
        //the index of this node in all leaf nodes
        if (this->nodes[i]->isLeaf){
            this->simplified_tree[i].leaf_index     = leaf_index;
            //during build_tree the leaf_index is not compute, do here.
            this->nodes[i]->leaf_index              = leaf_index;
            leaf_index++;
        }
    }

}

template<typename T>
int  Convex_Tree<T>::build_tree(int cur_depth, Vector<T>&alpha,  T beta, int parent_node_index,
                                Vector<int>& index_vec, Vector<T> cur_center)
{
    int rows=0;

    int cols= m_data.ncols();
    //Vector<int> indexes(1);
    ConvexNode<T>* cur_node=new ConvexNode<T>();
    this->nodes.push_back(cur_node);

    //current index recorder
    Vector<int> cur_index_vec;
    if (this->nodes_num==0){
        rows= m_data.nrows();
        cur_index_vec= createAnIndexedVector<int>(rows);
        //cur_node->ALPHA=NULL;
        //cur_node->BETA=NULL;
        cur_node->index=nodes_num;
        nodes_num++;
        cur_node->parent_index=-1;
        cur_node->ancestor_nodes_num=0;
        cur_node->ancestor_node_ids=NULL;
        //cur_node->center=NULL;
        cur_node->left=-1;
        cur_node->right=-1;
        //cur_node->left_center=NULL;
        //cur_node->right_center=NULL;
        cur_node->isLeaf=false;
    }else{
        rows= index_vec.size();
        cur_index_vec=index_vec;
        cur_node->index=nodes_num;
        cur_node->center=cur_center;
        this->nodes_num++;

        cur_node->parent_index=parent_node_index;

        //---------------------------------For each node save its ancestor chain---------------------------//
        int tmp_num=nodes[parent_node_index]->ancestor_nodes_num;
        cur_node->ancestor_nodes_num=tmp_num+1;
        cur_node->ancestor_node_ids=new int[cur_node->ancestor_nodes_num];
        if (nodes[parent_node_index]->ancestor_node_ids!=NULL){
            memcpy(cur_node->ancestor_node_ids, nodes[parent_node_index]->ancestor_node_ids, sizeof(int)*tmp_num);
        }
        cur_node->ancestor_node_ids[tmp_num]=parent_node_index;
        //------------------------------------------------------------------------------------------------//

        int alpha_num=this->nodes[parent_node_index]->ALPHA.nrows();
        cur_node->ALPHA.resize(alpha_num+1,cols);
        cur_node->BETA.resize(alpha_num+1);

        //for each node, it should shift its parent's constraints in order to shrink the convex scope
        for (int i=0;i<alpha_num;i++){
            Vector<T> alpha_vec(this->nodes[parent_node_index]->ALPHA[i],cols);
            T new_beta= this->getBeta_1(alpha_vec,nodes[parent_node_index]->BETA[i],cur_index_vec);
            cur_node->ALPHA.setRow(i,this->nodes[parent_node_index]->ALPHA[i]);
            //shift
            cur_node->BETA[i]=new_beta;
        }
        cur_node->ALPHA.setRow(alpha_num,alpha);
        cur_node->ALPHA_t= cur_node->ALPHA.transform_matrix();
        //print_matrix("ALPHA_t=",cur_node->ALPHA_t);
        //print_matrix("ALPHA=",cur_node->ALPHA);
        cur_node->BETA[alpha_num]=beta;
    }


    /*-------------------------------------------leaf node-------------------------------------------------*/
    if (rows<=this->leaf_pts_num_threshold){
        cur_node->isLeaf=true;
        cur_node->pts_indexes=cur_index_vec;
        cur_node->data=this->m_data.extractRows(cur_index_vec);
        this->leaf_nodes_num++;
        cur_node->pts_number=cur_node->data.nrows();
        //cur_node->print();

        return cur_node->index;
    }
    /*-------------------------------------------leaf node-------------------------------------------------*/

    /*---------------------------------------non-leaf node-------------------------------------------------*/
    int tmp_index = rand()% cur_index_vec.size();
    Vector<T> tmpRow=m_data.extractRow(cur_index_vec[tmp_index]);
    Vector<T> dists= pdist2(tmpRow, this->m_data,cur_index_vec);
    //select two point that are far away each other
    int left_index= index_max(dists);

    left_index= index_max(dists);
    dists.destroy();
    int real_left_index= cur_index_vec[left_index];

    Vector<T> left_center=this->m_data.extractRow(real_left_index);
    Vector<T> dists_to_left= pdist2( left_center,this->m_data,cur_index_vec);
    //print_vector("dists_to_left:",dists_to_left);
    int right_index= index_max(dists_to_left);
    int real_right_index=cur_index_vec[right_index];
    Vector<FLOAT_TYPE> right_center=this->m_data.extractRow(real_right_index);

    Vector<T> dists_to_right= pdist2( right_center,this->m_data,cur_index_vec);
    //print_vector("dists_to_left:",dists_to_left);
    //print_vector("dists_to_right:",dists_to_right);
    Vector<T> dists_diff= dists_to_left-dists_to_right;
    dists_to_left.destroy();
    dists_to_right.destroy();
    //print_vector("dists_diff:",dists_diff);
    Vector<int> assigned_to_left_indexes= dists_diff.findLessAndEqual(0);
    Vector<int> assigned_to_real_left_indexes= cur_index_vec.extract(assigned_to_left_indexes);
    assigned_to_left_indexes.destroy();
    //Matrix<T> assigned_to_left_pts=this->m_data.extractRows(assigned_to_real_left_indexes);
    //Vector<int> new_left_indexes_vec= cur_index_vec.extract(assigned_to_left_indexes);

    Vector<int> assigned_to_right_indexes= dists_diff.findLarge(0);
    //Vector<int> new_right_indexes_vec= cur_index_vec.extract(assigned_to_right_indexes);
    Vector<int> assigned_to_real_right_indexes= cur_index_vec.extract(assigned_to_right_indexes);
    assigned_to_right_indexes.destroy();


    //---mean point is the center of left_center and right_center
    Vector<FLOAT_TYPE> mean_point= left_center;
    mean_point+=right_center;
    mean_point/=2;

    Vector<FLOAT_TYPE> left_alpha= left_center-right_center;
    //normalize it
    left_alpha /=norm_v(left_alpha);

    /*-------------------------------------------build left--------------------------------------------------*/
    if (assigned_to_real_left_indexes.size()>0){
        FLOAT_TYPE left_beta= this->getBeta(left_alpha,mean_point,assigned_to_real_left_indexes);
        cur_depth++;
        if (cur_depth>this->depth)
        this->depth++;
        int left_child_index =this->build_tree(cur_depth,left_alpha,left_beta,cur_node->index,assigned_to_real_left_indexes,left_center);
        cur_node->left= left_child_index;
        cur_node->left_center= left_center;
    }


    /*-------------------------------------------build right--------------------------------------------------*/
    if (assigned_to_real_right_indexes.size()>0){
        Vector<FLOAT_TYPE> right_alpha= -left_alpha;
        FLOAT_TYPE right_beta= getBeta(right_alpha,mean_point,assigned_to_real_right_indexes);
        mean_point.destroy();
        int right_child_index=this->build_tree(cur_depth, right_alpha,right_beta,cur_node->index,assigned_to_real_right_indexes,right_center);
        cur_node->right=right_child_index;
        cur_node->right_center= right_center;
    }

    return cur_node->index;
}


//check whether all points of a node are inside the node
template <typename T>
bool Convex_Tree<T>::check_node_valid(int check_node_index){
    return check_sub_nodes_valid(check_node_index,check_node_index);
}

//find the nearest neighbor in nodes[node_index]
template <typename T>
FLOAT_TYPE Convex_Tree<T>::find_nn_in_node(int node_index, Vector<T>& q){
    FLOAT_TYPE result=0;
    if (!this->nodes[node_index]->isLeaf){
        int left_node=this->nodes[node_index]->left;
        FLOAT_TYPE result_left= find_nn_in_node(left_node,q);
        //if (!result)
        //    return false;

        int right_node=this->nodes[node_index]->right;
        FLOAT_TYPE result_right=find_nn_in_node(right_node,q);

        result=result_left ;
        if (result> result_right)
            result=result_right ;
    }else{
        Vector<FLOAT_TYPE> dists_squre= pdist2_squre(this->nodes[node_index]->data,q);
        int min_loc=index_min(dists_squre);
        result= dists_squre[min_loc];
    }
    return result;
}

//find the node where a point locates.
template <typename T>
int Convex_Tree<T>::find_node_index_for_a_point(int point_index){
    for (int i=0;i<this->leaf_nodes_num;i++){
        int ori_node_index= this->leaf_nodes_ori_indexes[i] ;
        int pts_num= this->nodes[ori_node_index]->pts_indexes.size();
        for (int j=0;j<pts_num;j++){
            if (point_index==this->nodes[ori_node_index]->pts_indexes[j]){
                return i;
            }
        }
    }

    //---no found
    return -1;
}


template <typename T>
bool Convex_Tree<T>::check_sub_nodes_valid(int check_node_index, int node_index){
    bool result=true;
    if (!this->nodes[node_index]->isLeaf){
        int left_node=this->nodes[node_index]->left;
        result= check_sub_nodes_valid(check_node_index, left_node);
        //if (!result)
        //    return false;

        int right_node=this->nodes[node_index]->right;
        result=check_sub_nodes_valid(check_node_index, right_node);
        //if (!result)
        //    return false;
    }else{
        int pts_num= this->nodes[node_index]->pts_number;
        for (int i=0;i<pts_num;i++){
            int pts_index= this->nodes[node_index]->pts_indexes[i];
            Vector<T> q= this->m_data.extractRow(pts_index);
            if (!this->nodes[check_node_index]->is_a_inner_point(q)){
                result=false;
                std::cout<<"check_node_index="<< check_node_index<<", leaf_node="<<node_index<<", pts_index=" <<pts_index<<", \n";
            }
        }
    }
    return result;
}


template <typename T>
bool Convex_Tree<T>:: check_all_leaf_nodes_valid(){
    bool result=true;
    for (int i=0;i<this->nodes.size();i++){
        if (this->nodes[i]->isLeaf)
            result=check_node_valid(i);
    }
    return result;
}

template <typename T>
void Convex_Tree<T>::print_tree_info()
{
    std::cout<<"Tree info: \n";
    std::cout<<"      data size= "<<this->m_data.nrows()<< ", dim="<< this->m_data.ncols()<<"\n";
    std::cout<<"      depth= "<<this->depth<<"\n";
    std::cout<<"      nodes num= "<<this->nodes.size()<<"\n";
    std::cout<<"      leaf nodes num="<< this->leaf_nodes_num<<"\n";
    std::cout<<"      percent threshold to be a leaf="<<this->leaf_pts_percent<<"\n";
    std::cout<<"      number threshold to be a leaf="<<this->leaf_pts_num_threshold<<"\n";
    std::cout<<"      tree building time: ";
    this->timer_build_tree.print();
}




template <typename T>
void Convex_Tree<T>::save_tree_info(FILE* log_file)
{
    char buffer[1000];
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer,"\n   data dim=");
    strcat(buffer,"Tree info: \n");
    strcat(buffer,"      data size= ");
    my_strcat(buffer,this->m_data.nrows());
    strcat(buffer,", dim=");
    my_strcat(buffer,this->m_data.ncols());
    strcat(buffer,"\n      depth= ");
    my_strcat(buffer,this->depth);
    strcat(buffer,"\n      nodes num= ");
    my_strcat(buffer,this->nodes.size());
    strcat(buffer,"\n      leaf nodes num=");
    my_strcat(buffer, this->leaf_nodes_num);
    strcat(buffer,"\n      percent threshold to be a leaf=");
    my_strcat_double(buffer,this->leaf_pts_percent);
    strcat(buffer,"\n      number threshold to be a leaf=");
    my_strcat(buffer,this->leaf_pts_num_threshold);
    strcat(buffer,"\n      tree building time: ");

    timer_build_tree.strcat_to_buffer(" ",buffer);
    //my_strcat(buffer,timer_build_tree.elapsed_time_total);
    strcat(buffer,"s\n");
    write_file_log(buffer, log_file);

}



//init current_visiting_node_indexes_of_query_points which set all current visiting nodes as the root.
template <typename T>
void Convex_Tree<T>::init_query_points_info(Matrix<T>& query_points_mat){
    int query_points_num= query_points_mat.nrows();
    query_points_vec.resize(query_points_num);
    for (int i=0;i<query_points_num;i++){
        query_points_vec[i]=query_points_mat.extractRow(i);
    }

    //create a query points vector, all query data are save into this->query_points_vec.
    //query_points_mat.write_data_into_vectors(this->query_points_vec);
}

//virtual procedure
template <typename T>
void Convex_Tree<T>::print_kNN_running_time_info()
{
 };

//virtual procedure
template <typename T>
void Convex_Tree<T>::save_KNN_running_time_info(FILE* log_file)
{
};

#endif // CONVEX_TREE_H_INCLUDED

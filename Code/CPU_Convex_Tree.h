#ifndef CPU_CONVEX_TREE_H
#define CPU_CONVEX_TREE_H

#include "Convex_Tree.h"
#include "ConvexNode.h"
#include "basic_functions.h"
#include "openCL_tool.h"
#include "cyw_timer.h"
#include "cyw_types.h"

//the CPU version of Convex_Tree
template <class T>
    class CPU_Convex_Tree :public Convex_Tree<T>
{
    public:
        CPU_Convex_Tree();
        virtual ~CPU_Convex_Tree(){
            //free  cur_k_NNResult
            for (int i=0; i<this->cur_k_NNResult.size();i++){
                this->cur_k_NNResult[i]->free_stru();
            }

            for (int i=0;i<this->query_points_nodes_alpha_scalar_product.size();i++){
                delete this->query_points_nodes_alpha_scalar_product[i];
            }
        };
        //leaf_pts_percent : the percent that specify the number of leaf_pts_num
        CPU_Convex_Tree(Matrix<T>& data, FLOAT_TYPE leaf_pts_percent, int alg_type);

        //after kNN processing, get the result by the following three entries
        //they are all virtual procedures, and override here.
        virtual FLOAT_TYPE*  get_kNN_dists(int query_point_index){
            return this->cur_k_NNResult[query_point_index]->k_dists;
        };
        virtual FLOAT_TYPE*  get_kNN_dists_squre(int query_point_index){
            return this->cur_k_NNResult[query_point_index]->k_dists_squre;
        };

        //override
        virtual  int* get_kNN_indexes(int query_point_index){
            return this->cur_k_NNResult[query_point_index]->k_indexes;
        };

        //override
        virtual void print_kNN_running_time_info();

    protected:
        //this is used for saving current k_NNResult
        Vector<NNResult<T>*> cur_k_NNResult;

        //override
        void virtual do_kNN(Matrix<T> &query_points_mat);

        /*
            This is virtual procedure, a entry for specific task in CPU_Convex_tree.
            it will be call in init_process_query_points defined in  superclass 'Convex_tree'
        */
        void virtual init_kNN_result();


        void virtual do_print_kNN_result(){
            int num_query_pts= this->cur_k_NNResult.size();
            for (int i=0;i<num_query_pts;i++){
                std::cout<<"\n i="<<i<<" ";
                this->cur_k_NNResult[i]->print();
            }
        }

        //return the approximate leave node index, it will be called in init_running_time_opencl_param
        int approximate_kNN( Vector<T>& q, int q_index);

    private:
        int appr_visiting_nodes_num=0;
        void kNN_cpu_by_leaf();
        void kNN_cpu_recurisive();
        NNResult<T>* NN_a_point_by_leaf(Vector<T> &q, int q_index);

        //---Saves all scalar product between query_points and the alpha of each node.
        //---for a node_i with k alphas, and its parent node_j with k-1 alphas,
        //---the first k-1 alphas in node_i is the same as node_j, only beta value is different.
        //---Therefore, in order to get the distances from q to all hyperplane in node_i, we only
        //---compute the last alpha(k)'*q +beta(k), because the first alpha'*q is already known.
        //---query_points_nodes_alpha_scalar_product[i]->v[j] saves the i^th scalar between query point
        //---and last alpha in node_j.
        Vector<Vector<FLOAT_TYPE>* > query_points_nodes_alpha_scalar_product;
        void do_kNN_recurisive(int cur_node_index, int query_point_index);

        //---the loop version of do_kNN_recurisive
        void do_kNN_recurisive_by_loop(int query_point_index);

        FLOAT_TYPE get_current_k_dists_squre(int query_point_index){
            return this->cur_k_NNResult[query_point_index]->get_kth_dist_square();
        }

        FLOAT_TYPE get_current_k_dists(int query_point_index){
            return this->cur_k_NNResult[query_point_index]->get_kth_dist();
        }


        Vector<NNResult<T>*>& get_cur_k_NNResult(){
            return this->cur_k_NNResult;
        }
};

template <class T>
CPU_Convex_Tree<T>::CPU_Convex_Tree(Matrix<T>& data,
                                    FLOAT_TYPE leaf_pts_percent,
                                    int alg_type):Convex_Tree<T>(data, leaf_pts_percent,alg_type)
{
}

//override
template <typename T>
void CPU_Convex_Tree<T>::init_kNN_result(){
    //cpu_convex_tree has nth to do here.
    int query_points_num=this->query_points_vec.size();

    //create cur_k_NNResult
    for (int i=0; i<this->cur_k_NNResult.size();i++)
    {
        this->cur_k_NNResult[i]->free_stru();
    }
    //this is much fast than above code.
    this->cur_k_NNResult=NNResult<T>::create_an_NNResult_vec(query_points_num, this->K);
    this->appr_leaf_node_indexes.resize(this->query_points_vec.size());
};


//ovreride do_kNN defined in super class
template <typename T>
void CPU_Convex_Tree<T>::do_kNN(Matrix<T> &query_points_mat){
    int q_num= query_points_mat.nrows();

    //---create a matrix and init as 0;
    //---query_points_nodes_alpha_scalar_product[i,j] saves the i^th scalar between query point
    this->query_points_nodes_alpha_scalar_product.resize(q_num);
    for (int i=0;i<q_num;i++){
        Vector<FLOAT_TYPE>* tmp_v= new Vector<FLOAT_TYPE>();

        //--all scalar product between query point and alpha in nodes are initialized as 0.
        tmp_v->resize(this->nodes_num);
        this->query_points_nodes_alpha_scalar_product[i]= tmp_v;
    }

    if (this->ALG_TYPE==USE_CPU_LEAF_APPRXIMATE_QP){
        kNN_cpu_by_leaf();
    }

    if (this->ALG_TYPE==USE_CPU_RECURSIVE_APPRXIMATE_QP){
        kNN_cpu_recurisive();
    }
}


//directly search leaves not from top to bottom
template <typename T>
void CPU_Convex_Tree<T>::kNN_cpu_by_leaf(){
    for (int i=0;i<this->query_points_vec.size();i++){
        Vector<T>& q= this->query_points_vec[i];
        NN_a_point_by_leaf(q, i);
    }
}


template <typename T>
void CPU_Convex_Tree<T>::kNN_cpu_recurisive(){
    for (int i=0;i<this->query_points_vec.size();i++){
        Vector<T>& q= this->query_points_vec[i];
        this->approximate_kNN(q,i);
        //do_kNN_recurisive(0, i);
        do_kNN_recurisive_by_loop(i);
    }
    //std::cout<<"Appr_visiting_nodes_num= "<< this->appr_visiting_nodes_num;
}




// return the approximate leave node index
template <typename T>
int CPU_Convex_Tree<T>::approximate_kNN(Vector<T> &q, int q_index)
{
    int cur_node_index=0;
    this->timer_total_approximate_searching.start_my_timer();
    while (this->nodes[cur_node_index]->isLeaf==false){
        //this->nodes[cur_node_index]->print()    ;
        FLOAT_TYPE left_dist_squre=pdist2_squre(q,this->nodes[cur_node_index]->left_center);

        FLOAT_TYPE right_dist_squre=pdist2_squre(q,this->nodes[cur_node_index]->right_center);
        this->dist_computation_times_in_host+=2;
        if (left_dist_squre>=right_dist_squre){
            cur_node_index=this->nodes[cur_node_index]->right;
        } else
            cur_node_index=this->nodes[cur_node_index]->left;
       this->appr_visiting_nodes_num++;
    };
    this->timer_total_approximate_searching.stop_my_timer();

    //obtain all distances from q to points in the final leaf
    //this->nodes[cur_node_index]->print();
    int data_len=this->nodes[cur_node_index]->data.nrows();
    this->timer_total_dists_computation.start_my_timer();
    Vector<FLOAT_TYPE> dists_square_v=pdist2_squre(q, this->nodes[cur_node_index]->data);
    this->timer_total_dists_computation.stop_my_timer();
    this->dist_computation_times_in_host+= this->nodes[cur_node_index]->pts_indexes.size();

    //find the k min dists, as well as indexes
    this->timer_total_update_kNN_dists.start_my_timer();
    for (int i=0;i<data_len;i++){
        int cur_index = this->nodes[cur_node_index]->pts_indexes[i];
        this->cur_k_NNResult[q_index]->update(dists_square_v[i],cur_index);
    }
    this->timer_total_update_kNN_dists.stop_my_timer();

    this->appr_leaf_node_indexes[q_index]=cur_node_index;
    return cur_node_index;
}

template <typename T>
void CPU_Convex_Tree<T>::do_kNN_recurisive(int cur_node_index, int query_point_index){
    Vector<T>&q= this->query_points_vec[query_point_index];

    if (!this->nodes[cur_node_index]->isLeaf)
    {
        int left_node= this->nodes[cur_node_index]->left;
        bool is_candidate=false;

        this->timer_total_quadprog.start_my_timer();
        //if the current node is not leaf
        //FLOAT_TYPE cur_kth_min_dist_squre=this->get_current_k_dists_squre(query_point_index);
        FLOAT_TYPE cur_kth_min_dist=this->get_current_k_dists(query_point_index);
        int tmp_scalar_product=0;
        is_candidate=!this->nodes[left_node]->is_appr_min_dist_from_q_larger_by_hyper_plane(q,cur_kth_min_dist,&tmp_scalar_product);

        //is_candidate=!this->nodes[left_node]->is_appr_min_dist_from_q_larger_by_hyper_plane_pro(q,cur_kth_min_dist,this->nodes,this->query_points_nodes_alpha_scalar_product[query_point_index],&tmp_scalar_product);
        this->scalar_product_num += tmp_scalar_product;

        this->timer_total_quadprog.stop_my_timer();
        this->quadprog_cal_num++;

        if(is_candidate)
        {
            do_kNN_recurisive( left_node,query_point_index);
        }

        int right_node= this->nodes[cur_node_index]->right;

        is_candidate=false;
        this->timer_total_quadprog.start_my_timer();
        //cur_kth_min_dist_squre=this->get_current_k_dists_squre(query_point_index);
        cur_kth_min_dist=this->get_current_k_dists(query_point_index);
        tmp_scalar_product=0;

        is_candidate=!this->nodes[right_node]->is_appr_min_dist_from_q_larger_by_hyper_plane(q,cur_kth_min_dist,&tmp_scalar_product);

        //is_candidate=!this->nodes[right_node]->is_appr_min_dist_from_q_larger_by_hyper_plane_pro(q,cur_kth_min_dist,this->nodes,
        //                                                                                         *this->query_points_nodes_alpha_scalar_product[query_point_index],
        //                                                                                         &tmp_scalar_product);
        this->scalar_product_num += tmp_scalar_product;

        this->timer_total_quadprog.stop_my_timer();
        this->quadprog_cal_num++;

        if(is_candidate)
        {
            do_kNN_recurisive( right_node,query_point_index);
        }
    }

    //if the current node is a leaf, and is not handled in approximateNN
    if (this->nodes[cur_node_index]->isLeaf&& (cur_node_index!=this->appr_leaf_node_indexes[query_point_index]))
    {
        //obtain all distances from q to points in the final leaf
        this->timer_total_dists_computation.start_my_timer();
        Vector<FLOAT_TYPE> dists_squre=pdist2_squre(this->nodes[cur_node_index]->data,q);
        this->timer_total_dists_computation.stop_my_timer();

        int data_len=this->nodes[cur_node_index]->pts_indexes.size();
        this->dist_computation_times_in_host+= data_len;

        this->timer_total_update_kNN_dists.start_my_timer();
        //update knn dists
        for (int j=0;j<data_len;j++){
            int ori_pts_index= this->nodes[cur_node_index]->pts_indexes[j];
            this->cur_k_NNResult[query_point_index]->update(dists_squre[j],ori_pts_index);
        }
        this->timer_total_update_kNN_dists.stop_my_timer();
    }
    return;
}


template <typename T>
void CPU_Convex_Tree<T>::do_kNN_recurisive_by_loop( int query_point_index){
    Vector<T>&q= this->query_points_vec[query_point_index];

    int visiting_times_recorder[this->nodes_num];

    for (int i=0; i<this->nodes_num;i++){
        visiting_times_recorder[i]=0;
    }
    int cur_node_index=0;

    //if a node has been visited 3 times, then it is handled including all its sub-nodes
    while (visiting_times_recorder[0]<3){
        visiting_times_recorder[cur_node_index]++;
        if (!this->nodes[cur_node_index]->isLeaf)
        {
            int next_node_index=0;
            if (visiting_times_recorder[cur_node_index]==1)
            {   //if it is the first time visiting the current non-leaf node, then check its left child
                next_node_index= this->nodes[cur_node_index]->left;
            }

            if (visiting_times_recorder[cur_node_index]==2)
            {   //if it is the second time visiting the current non-leaf node, then check its right child
                next_node_index= this->nodes[cur_node_index]->right;
            }

            if (visiting_times_recorder[cur_node_index]==3)
            {   //if it is the third time visiting the current non-leaf node, which means both its left
                //sub-nodes and ritht sub-nodes are all handled, then go back to it's parent.
                cur_node_index= this->nodes[cur_node_index]->parent_index;
                //next loop, skip the following process.
                continue;
            }

            bool is_candidate_node=false;
            int constrains_num= this->nodes[next_node_index]->ALPHA.nrows()   ;  //all_constrains_num_of_each_nodes[next_node_index];

            FLOAT_TYPE cur_kth_min_dist=this->get_current_k_dists(query_point_index);

            int tmp_scalar_product=0;

            this->timer_total_quadprog.start_my_timer();
            //is_candidate_node =!this->nodes[next_node_index]->is_appr_min_dist_from_q_larger_by_hyper_plane(q,cur_kth_min_dist,&tmp_scalar_product);
            this->quadprog_cal_num++;

            is_candidate_node=!this->nodes[next_node_index]->is_appr_min_dist_from_q_larger_by_hyper_plane_pro(q,
                                                                                                             cur_kth_min_dist,
                                                                                                             *this->query_points_nodes_alpha_scalar_product[query_point_index],
                                                                                                             &tmp_scalar_product);
            this->scalar_product_num += tmp_scalar_product;
            this->timer_total_quadprog.stop_my_timer();
            //if left node is within the scope, then go to next node
            if(is_candidate_node)
            {
                cur_node_index=next_node_index;
            }
        }

        if ( this->nodes[cur_node_index]->isLeaf)
        {
            if (cur_node_index!=this->appr_leaf_node_indexes[query_point_index]){
                  //obtain all distances from q to points in the final leaf
                this->timer_total_dists_computation.start_my_timer();
                Vector<FLOAT_TYPE> dists_squre=pdist2_squre(this->nodes[cur_node_index]->data,q);
                this->timer_total_dists_computation.stop_my_timer();

                int data_len=this->nodes[cur_node_index]->pts_indexes.size();
                this->dist_computation_times_in_host+= data_len;

                this->timer_total_update_kNN_dists.start_my_timer();
                //update knn dists
                for (int j=0;j<data_len;j++){
                    int ori_pts_index= this->nodes[cur_node_index]->pts_indexes[j];
                    this->cur_k_NNResult[query_point_index]->update(dists_squre[j],ori_pts_index);
                }
                this->timer_total_update_kNN_dists.stop_my_timer();
            }
            cur_node_index= this->nodes[cur_node_index]->parent_index;

        }
    }//end while

}

//directly search leaves not from top to bottom
template <typename T>
NNResult<T>* CPU_Convex_Tree<T>::NN_a_point_by_leaf(Vector<T> &q, int q_index)
{
    this->approximate_kNN(q,q_index);

    register int i, j, nodes_num=this->nodes.size();

    for (i=0; i<nodes_num;i++)
    {
        if (this->nodes[i]->isLeaf)
        {
            //inner node is handled in approximateBNN
            if (this->appr_leaf_node_indexes[q_index]==i)
                continue;
            this->timer_total_quadprog.start_my_timer();

            bool is_candidate=false;
            double cur_kth_min_dist=  this->cur_k_NNResult[q_index]->get_kth_dist();
            int tmp_scalar_product=0;
            is_candidate=!this->nodes[i]->is_appr_min_dist_from_q_larger_by_hyper_plane(q,cur_kth_min_dist,&tmp_scalar_product);

            //is_candidate=!this->nodes[i]->is_appr_min_dist_from_q_larger_by_hyper_plane_pro(q,cur_kth_min_dist,this->nodes,*this->query_points_nodes_alpha_scalar_product[q_index],&tmp_scalar_product);
            this->scalar_product_num += tmp_scalar_product;
            this->timer_total_quadprog.stop_my_timer();
            this->quadprog_cal_num++;

            if (is_candidate)
            {
                int leaf_index=this->nodes[i]->leaf_index;
                //obtain all distances from q to points in the final leaf
                this->timer_total_dists_computation.start_my_timer();
                int offset=this->leaf_nodes_start_pos_in_data_set[leaf_index];
                int data_len=this->nodes[i]->data.nrows();
                //compute the distances
                pdist2_squre(this->nodes[i]->data,q, this->m_dists_square_v,offset);
                Vector<FLOAT_TYPE>& dists_squre=this->m_dists_square_v;
                this->timer_total_dists_computation.stop_my_timer();
                this->dist_computation_times_in_host+= this->nodes[i]->pts_indexes.size();

                this->timer_total_update_kNN_dists.start_my_timer();
                //update knn dists
                for (j=offset; j<offset+data_len;j++ ){
                    int ori_pts_index= this->nodes[i]->pts_indexes[j-offset];
                    this->cur_k_NNResult[q_index]->update(dists_squre[j],ori_pts_index);
                }
                this->timer_total_update_kNN_dists.stop_my_timer();
            }
        }
    }

    return this->cur_k_NNResult[q_index];
}


//override virtual procedure
template <typename T>
void CPU_Convex_Tree<T>::print_kNN_running_time_info()
{
    std::cout<<"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n";
    std::cout<<"*     Query points number:"<<this->query_points_vec.size()<<", and K="<<this->K<<"\n";
    std::cout<<"*     CPU Alorithm ";
    this->timer_whole_alg.print();
    std::cout<<"*          Where:\n";
    this->timer_init_process_query_points.print("*          (1) Init query points, ");
    std::cout<<"*          (2) Distance computations number="<< this->dist_computation_times_in_host<<", ";
    this->timer_total_dists_computation.print();
    std::cout<<"*          (3) Quadratic programming number= "<< this->quadprog_cal_num<<",\n";
    std::cout<<"*              where total scalar product number= "<< this->scalar_product_num<<", ";
    this->timer_total_quadprog.print();
    this->timer_total_approximate_searching.print("*          (4) Approximate searching, ");
    this->timer_total_update_kNN_dists.print("*          (5) Update dist time");
    std::cout<<"*-----------------------------------------KNN QUERY RESULT----------------------------------------";

};


#endif // CPU_CONVEX_TREE_H

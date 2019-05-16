/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#ifndef ConvexNode_H
#define ConvexNode_H

#include "Array.hh"
#include "basic_functions.h"
#include "cyw_types.h"
template<typename T>
struct ConvexNode
{
    //isLeaf=1 means this node is a leaf, otherwise it is a node
    bool isLeaf;
    //if node is a leaf, then data saves the points within it.
    Matrix<T> data;

    //the index of the node creating in all nodes, also it is the order of traversing the whole tree from top to down
    int index=-1;
    int parent_index=-1;

    int ancestor_nodes_num;
    int *ancestor_node_ids;

    //the index of the node in all leaves nodes, if the node is not a leaf leave_index=-1
    int leaf_index=-1;

    //the off_set of the data saved in sorted_data in bkm tree
    //the data of the first leave node start from 0 and end at first_leave_node.data.length()-1,
    //the data of the second leave node start from first_leave_node.data.length(), and
    //end at first_leave_node.data.length()+ second_leave_node.data.length()-1
    //....
    int saved_data_offset=0;

    //two children
    int left=-1;
    int right=-1;

    int pts_number=-1;
    Vector<int> pts_indexes;

    //ALPHA and BETA are two sets that each alpha_i \in ALPHA and beta_i \in Beta determine a hyper plane
    //such that for all point x belongs to this nodes satisfy "alpha_i'*x +beta_i>=0;"
    //In fact, suppose there is n alpha in node_i, the first (n-1) alpha is the same as node_j which is
    //the parent node of node_i, the reason we save all alpha in a node is to improve the efficiency of
    //computing in approximate_min_dist_by_hyper_plane, or else we have to go back to its all ancestor nodes.
    Matrix<FLOAT_TYPE> ALPHA;
    //the transform of ALPHA, i.e. ALPHA=ALPHA_t'
    Matrix<FLOAT_TYPE> ALPHA_t;


    Vector<FLOAT_TYPE> BETA;

    //current geometrical center of this node
    Vector<FLOAT_TYPE> center;
    //two children's geometrical centers of this node
    Vector<FLOAT_TYPE> left_center;
    Vector<FLOAT_TYPE> right_center;

    void save_to_file(){
    }

    //return true if q is a inner point
    bool is_a_inner_point( Vector<T>& q){
        Vector<T> check_v= this->ALPHA.product(q);
        check_v +=this->BETA;
        int len= BETA.size();
        for (int i=0;i<len; i++)
        {
            // if there exists a alpha and beta such that alpha[i]'* point +beta[i]<0
            // point is not in the convex set
            if (check_v[i]<0)
                if (abs(check_v[i])>10e-10)
                   return false;
        }
        return true;
    }



    //retrun a approximate min dist from q to this convex node
    //idea: if a point q is outside of this node, then max distance from
    //      q to each active constrains (hyperplanes) is the approximate
    //      distance. Because the \forall active hyperplane h, we have
    //      d_min >= dist(q,h), i.e d_min>= max(q,h) for all active h;
    FLOAT_TYPE approximate_min_dist_by_hyper_plane( Vector<T>& query_point){
        Vector<T> check_v= this->ALPHA.product(query_point);
        check_v +=this->BETA;
        int len= BETA.size();
        FLOAT_TYPE result=0;
        for (int i=0;i<len; i++)
        {
            // if there exists a alpha and beta such that alpha[i]'* point +beta[i]<0
            // point is not in the convex set
            if (check_v[i]<0){
                //abs(check_v[i]) is the dist from q to current hyperplane.
                if (result < abs(check_v[i])){
                    result = abs(check_v[i]);
                }
            }
        }
        return result;
    }


    //return true if the d_min from q to this node is larger than dist_compare.
    bool is_appr_min_dist_from_q_larger_by_hyper_plane( Vector<T>& query_point,
                                                        FLOAT_TYPE dist_compare,
                                                        int *scalar_product_num){
        /*
            //The following two line is checking my approximate_min_dist_by_hyper_plane(),
            //but it is relatively slow, for it compute all hyperplane, which is unnecessary.
            FLOAT_TYPE appr_dist= approximate_min_dist_by_hyper_plane(query_point);
            bool result = (appr_dist>dist_compare);
            return result;
        */
        int  product_num=0;
        bool result=false;
        int dim=query_point.size();
        FLOAT_TYPE* alpha_set=ALPHA.get_matrix_raw_data();

        for (int i=0;i<ALPHA.nrows();i++){
            FLOAT_TYPE  tmp_dist=this->BETA[i];
            for (int j=0;j<dim;j++){
                tmp_dist += alpha_set[i*dim+j]*query_point[j];
            }
            product_num++;
            if (tmp_dist<0){
                if (dist_compare < abs(tmp_dist)){
                    //if there exists one such hyper plane then return.
                    result=true;
                    break;
                }
            }
        }
        *scalar_product_num =product_num;
        return result;
    }


    //return true if the d_min from q to this node is larger than dist_compare.
    //scalar_product_num returns the number of scalar products.
    //but this procedure seems slower than is_appr_min_dist_from_q_larger_by_hyper_plane, because
    //it visits all its ancestor.
    bool is_appr_min_dist_from_q_larger_by_hyper_plane_pro( Vector<T>& query_point,
                                                            FLOAT_TYPE dist_compare,
                                                            Vector<FLOAT_TYPE>& query_point_scalar_product_from_all_nodes,
                                                            int *scalar_product_num){
        /*
            //The following two line is checking my approximate_min_dist_by_hyper_plane(),
            //but it is relatively slow, for it compute all hyperplane, which is unnecessary.
            FLOAT_TYPE appr_dist= approximate_min_dist_by_hyper_plane(query_point);
            bool result = (appr_dist>dist_compare);
            return result;
        */
        int  product_num=0;
        bool result=false;
        int dim=query_point.size();
        FLOAT_TYPE* alpha_set=ALPHA.get_matrix_raw_data();

        //visits all it ancestors, except root node
        int alpha_num=ALPHA.nrows();
        int cur_ancestor_node_id=0;

        for (int i=0;i<ALPHA.nrows();i++){
            FLOAT_TYPE  tmp_dist=this->BETA[i];

            if (i<ALPHA.nrows()-1)
                cur_ancestor_node_id=this->ancestor_node_ids[i+1];
            else
                cur_ancestor_node_id=this->index;

            bool need_compute =true;
            need_compute=(query_point_scalar_product_from_all_nodes[cur_ancestor_node_id]==0);

            //---if the scalar product is still not compute then compute it
            if (need_compute){
                FLOAT_TYPE tmp_product=0;
                for (int j=0;j<dim;j++){
                    tmp_product += alpha_set[i*dim+j]*query_point[j];
                }
                product_num++;
                query_point_scalar_product_from_all_nodes[cur_ancestor_node_id]=tmp_product;
                tmp_dist+=tmp_product;
            }else{
                //---else unnecessary to compute product
                tmp_dist+=query_point_scalar_product_from_all_nodes[cur_ancestor_node_id];
            }

            if (tmp_dist<0){
                if (dist_compare < abs(tmp_dist)){
                    //if there exists one such hyper plane then return.
                    result=true;
                    break;
                }
            }
        }
        *scalar_product_num =product_num;
        return result;
    }


    //count how many active constrains if query points is out of this node
    int count_active_constrains(Vector<FLOAT_TYPE>& query_points){
        if (this->ALPHA.nrows()==0)
            return 0;
        Vector<T> check_v= this->ALPHA.product(query_points);
        check_v +=this->BETA;
        int len= BETA.size();
        int result=0;
        for (int i=0;i<len; i++)
        {
            // if there exists a alpha and beta such that alpha[i]'* point +beta[i]<0
            // point is not in the convex set
            if (check_v[i]<0)
                result++;
        }
        return result;
    }

    void print(){
        std::cout<< "isLeaf:"<< isLeaf<<"\n";
        std::cout<< "node index:"<< index<<"\n";
        std::cout<< "parent index:"<< parent_index<<"\n";
        std::cout<< "left index, right index:"<< left<<", "<< right<<"\n";
        print_vector("pts_index:", pts_indexes);
        print_matrix("ALPHA:", ALPHA);
        print_vector("BETA:", BETA);
        print_vector("center:", center);
        print_vector("left_center:", left_center);
        print_vector("right_center:", right_center);
    }
};

// this is a simplified convex_node structure of ConvexNode, and is used as
// a parameter that can be passed to opencCL device directly.
typedef struct Simple_Convex_Node {
     bool isLeaf;
     int  node_index;
     int  parent_index;
     int  leaf_index;       // the leaf index of this node in all leaf nodes
     int  left_node;
     int  right_node;
} CONVEX_TREE;
#endif // ConvexNode_H

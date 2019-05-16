/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#if USE_DOUBLE > 0
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FLOAT_TYPE double
#define FLOAT_TYPE4 double4
#define MAX_FLOAT_TYPE      1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define FLOAT_TYPE4 float4
#define FLOAT_TYPE8 float8
#define MAX_FLOAT_TYPE      3.402823466e+38f
#define MIN_FLOAT_TYPE     -3.402823466e+38f
#endif
#define VECSIZE 4
#define VECSIZE_8 8

    typedef struct Convex_Node {
         bool isLeaf;
         int  node_index;
         int  parent_index;
         int  leaf_index;       // the leaf index of this node in all leaf nodes
         int  left_node;
         int  right_node;
    } CONVEX_TREE;


    //the inner product of q and p
    FLOAT_TYPE scalar_product(FLOAT_TYPE* p, FLOAT_TYPE* q){
        //DIM will be written in "make_kernel_from_file"
        FLOAT_TYPE result=0;
        for (int i=0;i<DIM;i++){
            result += p[i]*q[i];
        }
        return result;
    }

    //use float4 to compute dist, it runs fast in GPU.
    FLOAT_TYPE float4_dist(__global FLOAT_TYPE *p,  FLOAT_TYPE *q){
        FLOAT_TYPE dist_tmp, tmp;
        FLOAT_TYPE4 dist_tmp_vec, tmp_vec;
        uint dim_mul_vecsize = VECSIZE * (DIM / VECSIZE);
        FLOAT_TYPE4 test_patt_vec[DIM / VECSIZE];
        int j;
        for (j=0; j<dim_mul_vecsize; j+=VECSIZE){
            test_patt_vec[j/VECSIZE] = (FLOAT_TYPE4)( p[j], p[j+1],  p[j+2], p[j+3] );
        }

        #if USE_DOUBLE > 0
                dist_tmp = 0.0;
                dist_tmp_vec = (FLOAT_TYPE4) (0.0,0.0,0.0,0.0);
        #else
                dist_tmp = 0.0f;
                dist_tmp_vec = (FLOAT_TYPE4) (0.0f,0.0f,0.0f,0.0f);
        #endif

        for (j=0; j<dim_mul_vecsize; j+=VECSIZE){
            tmp_vec = (FLOAT_TYPE4) (q[j+0],  q[j+1], q[j+2], q[j+3]);
            tmp_vec = tmp_vec - test_patt_vec[j/VECSIZE];
            dist_tmp_vec += tmp_vec*tmp_vec;
        }

        //remain dimensions
        for (j=dim_mul_vecsize; j<DIM; j++){
            tmp = (q[j] - p[j]);
            dist_tmp += tmp*tmp;
        }
        dist_tmp += dist_tmp_vec.s0 + dist_tmp_vec.s1 + dist_tmp_vec.s2 + dist_tmp_vec.s3;
        return dist_tmp;
    }

    //use float8 to compute dist, it runs fast in GPU.
    FLOAT_TYPE float8_dist(__global FLOAT_TYPE *p,  FLOAT_TYPE *q){
        FLOAT_TYPE dist_tmp, tmp;
        FLOAT_TYPE8 dist_tmp_vec, tmp_vec;
        uint dim_mul_vecsize = VECSIZE_8 * (DIM / VECSIZE_8);
        FLOAT_TYPE8 test_patt_vec[DIM / VECSIZE_8];
        int j;
        for (j=0; j<dim_mul_vecsize; j+=VECSIZE_8){
            test_patt_vec[j/VECSIZE_8] = (FLOAT_TYPE8)( p[j], p[j+1],  p[j+2], p[j+3], p[j+4], p[j+5],  p[j+6], p[j+7] );
        }

        #if USE_DOUBLE > 0
                dist_tmp = 0.0;
                dist_tmp_vec = (FLOAT_TYPE8) (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        #else
                dist_tmp = 0.0f;
                dist_tmp_vec = (FLOAT_TYPE8) (0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        #endif

        for (j=0; j<dim_mul_vecsize; j+=VECSIZE_8){
            tmp_vec = (FLOAT_TYPE8) (q[j+0],  q[j+1], q[j+2], q[j+3],q[j+4],  q[j+5], q[j+6], q[j+7]);
            tmp_vec = tmp_vec - test_patt_vec[j/VECSIZE_8];
            dist_tmp_vec += tmp_vec*tmp_vec;
        }

        //remain dimensions
        for (j=dim_mul_vecsize; j<DIM; j++){
            tmp = (q[j] - p[j]);
            dist_tmp += tmp*tmp;
        }
        dist_tmp += dist_tmp_vec.s0 + dist_tmp_vec.s1 + dist_tmp_vec.s2 + dist_tmp_vec.s3
                   +dist_tmp_vec.s4 + dist_tmp_vec.s5 + dist_tmp_vec.s6 + dist_tmp_vec.s7;
        return dist_tmp;
    }

    //retrun a approximate min dist from q to this convex node
    //d_min is still approximate distance, it can be improved or optimized.
    //idea: if a point q is outside of this node, then max distance from
    //      q to each active constrains (hyperplanes) is the approximate
    //      distance. Because the \forall active hyperplane h, we have
    //      d_min >= dist(q,h);
    FLOAT_TYPE approximate_min_dist_by_hyper_plane( FLOAT_TYPE* query_point,
                                                    FLOAT_TYPE* ALPHA,
                                                    FLOAT_TYPE* BETA,
                                                    int         ALPPHA_size){
        FLOAT_TYPE result=0;
        FLOAT_TYPE tmp_val=0;
        for (int i=0;i<ALPPHA_size; i++)
        {
            //DIM will be written in "make_kernel_from_file"
            FLOAT_TYPE* alpha= ALPHA+i*DIM;
            FLOAT_TYPE beta = BETA[i];
            tmp_val= scalar_product(alpha, query_point);
            // if there exists a alpha and beta such that alpha[i]'* point +beta[i]<0
            // point is not in the node
            if (tmp_val<0){
                if (result < -tmp_val){
                    result = -tmp_val;
                }
            }
        }
        return result;
    }




    //return true if the d_min from q to this node is larger than dist_compare.
    //@param dist_compute_times_in_appr_quadprog: count the dist computation times here
    bool is_appr_min_dist_from_q_larger_by_hyper_plane(          FLOAT_TYPE  *query_point,
                                                        __global FLOAT_TYPE  *ALPHA,
                                                        __global FLOAT_TYPE  *BETA,
                                                                        int   ALPPHA_size,
                                                                 FLOAT_TYPE   dist_compare,
                                                                       long  *dist_compute_times_in_appr_quadprog,
                                                                 FLOAT_TYPE  *query_point_scalar_product_from_all_nodes,
                                                               __global int  *cur_ancestor_nodes_ids)
    {
        bool result=false;
        int tmp_times=0;
        int cur_ancestor_node_id=0;
        for (int i=0;i<ALPPHA_size;i++){
            FLOAT_TYPE  tmp_dist=BETA[i];
            if (DIM<89){
                //---ORIGINAL SCALAR PRODUCT, MANY DUPLICATION, but, in low dim it is faster.
                for (int j=0;j<DIM;j++){
                    tmp_dist += ALPHA[i*DIM+j]*query_point[j];
                }
                tmp_times++;
            }else{
                //---FILTER UNNCESSARY SCALAR PRODUCT, BUT IT IS SLOW in low dimension.
                //---THE REASON IS VISISITING query_point_scalar_product_from_all_nodes RANDOMLY
                //---will cause pointer jumping, which IS SLOW.
                cur_ancestor_node_id=cur_ancestor_nodes_ids[i];

                bool need_compute =true;
                need_compute=(query_point_scalar_product_from_all_nodes[cur_ancestor_node_id]==0);
                //---if the scalar product is still not compute then compute it
                if (need_compute){
                    FLOAT_TYPE tmp_product=0;
                    for (int j=0;j<DIM;j++){
                        tmp_product += ALPHA[i*DIM+j]*query_point[j];
                    }
                    tmp_times++;
                    query_point_scalar_product_from_all_nodes[cur_ancestor_node_id]=tmp_product;
                    tmp_dist+=tmp_product;

                }else{
                    //---else unnecessary to compute product
                    tmp_dist+=query_point_scalar_product_from_all_nodes[cur_ancestor_node_id];
                }
            }//if (DIM<...)

            if (tmp_dist<0){
                if (dist_compare <= (tmp_dist*tmp_dist)){
                    //if there exists one such hyper plane then return.
                    result=true;
                    break;
                }
            }

        }
        *dist_compute_times_in_appr_quadprog+=tmp_times;
        return result;
    }


    //brute force computing and update dist_k_mins_private_tmp and idx_k_mins_global_tmp
    /*
        pts_num: the number of points in all_sorted_data_set.
    */
    void do_brute_force_and_update_private(          FLOAT_TYPE        *cur_query_point,
                                                            int         cur_query_point_index,
                                                            int         pts_num,
                                                            int         cur_leaf_node_start_pos,
                                            __global FLOAT_TYPE        *all_sorted_data_set,
                                            __global        int        *sorted_data_set_indexes,
                                                     FLOAT_TYPE        *dist_k_mins_private_tmp,
                                                            int        *idx_k_mins_private_tmp,
                                                            int         K_NN)
    {
        FLOAT_TYPE dist_squre_tmp=0;
        FLOAT_TYPE tmp=0;
        int tmp_idx=0;

        for (int i=0;i<pts_num;i++){
            /*-------------------------original dist computation-------------------------------------
                dist_squre_tmp=0;
                for (int j=0;j<DIM;j++){
                    tmp= all_sorted_data_set[(cur_leaf_node_start_pos+i)*DIM+j]-cur_query_point[j];
                    dist_squre_tmp+=tmp*tmp;
                }
                printf("\n ori dist_squre_tmp=%f", dist_squre_tmp);
            -------------------------original dist computation--------------------------------------*/

            dist_squre_tmp=0;
            //---use float4 to compute dist. It seems do not speedup for float type in CPU.
            dist_squre_tmp= float4_dist(all_sorted_data_set+ (cur_leaf_node_start_pos+i)*DIM, cur_query_point);

            //---use float8 to compute dist. It seems do not speedup for float type in CPU.
            //---dist_squre_tmp= float8_dist(all_sorted_data_set+ (cur_leaf_node_start_pos+i)*DIM, cur_query_point);

            //get the current k^th min_dist_square of current query point
            FLOAT_TYPE cur_k_min_dist_square=dist_k_mins_private_tmp[K_NN-1];

            if (cur_k_min_dist_square> dist_squre_tmp){
                //printf("update dist_k_mins_private_tmp...\n");
                //printf("cur_k_min_dist_square=%f,  dist_squre_tmp=%f \n",cur_k_min_dist_square,dist_squre_tmp );
                int j = K_NN - 1;
                dist_k_mins_private_tmp[j] = dist_squre_tmp;
                int pts_idx =sorted_data_set_indexes[cur_leaf_node_start_pos+i];
                idx_k_mins_private_tmp[j] = pts_idx;
                for(;j>0;j--){
                    if(dist_k_mins_private_tmp[j-1] > dist_k_mins_private_tmp[j]){
                        //printf("new nn found, swap...");
                        tmp=dist_k_mins_private_tmp[j-1];
                        dist_k_mins_private_tmp[j-1]=dist_k_mins_private_tmp[j];
                        dist_k_mins_private_tmp[j]  =tmp;

                        //swap indices
                        tmp_idx=idx_k_mins_private_tmp[j-1];
                        idx_k_mins_private_tmp[j-1]=idx_k_mins_private_tmp[j];
                        idx_k_mins_private_tmp[j]  =tmp_idx;
                    } else break;
                }
            }
        }
    }


/*
    0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                          is too large, we can submit subset of query sets to this kernel.
    1 candidate_query_points_indexes    : the indexes of current query points in all query points set
    2 candidate_query_points_set        : the current query points data set
    3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
    4 all_sorted_data_set               : all sorted data
    5 sorted_data_set_indexes           : all points indexes in sorted data set
    6 tree_struct                       : the tree structure of the whole tree. It is not used now.
    7 all_leaf_nodes_ALPHA_set          : ALPHA set of all leaf nodes
    8 leaf_nodes_BETA_set               : BETA set of all leaf nodes
    9 all_constrains_num_of_each_leaf_nodes  : all_constrains_num_of_each_nodes[i]=j means i^th leaf nodes has j constrains, i.e. has j alphas and betas
   10 all_leaf_nodes_offsets_in_all_ALPHA    : the offset of each leaf node in ALPHA
   11 leaf_node_num                     : the number of leaf nodes
   12 all_leaf_nodes_ancestor_nodes_ids : the ancestor nodes ids of each leaf nodes
   13 leaf_nodes_start_pos_in_sorted_data_set : specify the start position from which each sorted leave node in sorted data set
   14 pts_num_in_sorted_leaf_nodes      : the length of points saved in each sorted leave node
   15 dist_k_mins_global_tmp            : the K min-distance of all query points,
                                          the length of dist_mins_global_tmp is K* query_points_size
   16 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
                                          the length of dist_mins_global_tmp is K* query_points_size
   17 K_NN                              : the value of K
   18 dist_computation_times_arr        : dist_computation_times_arr[i] saves total distance computation times of the i^th point and
   19 quadprog_times_arr                : quadprog_times_arr[i] approximate quadprog times of the i^th point.
   20 dist_computation_times_in_quadprog: dist_computation_times_in_quadprog[i] saves the total distance computation times
                                          in quadprog of the i^th point.

*/
__kernel void do_finding_KNN_by_leaf_order(
                                        int          candidate_query_points_num,
                      const  __global        int         *candidate_query_points_indexes,
                      const  __global FLOAT_TYPE         *candidate_query_points_set,
                      const  __global        int         *candidate_query_points_appr_leaf_node_indexes,
                      const  __global FLOAT_TYPE         *all_sorted_data_set,
                      const  __global        int         *sorted_data_set_indexes,
                      const __global  CONVEX_TREE         *tree_struct,
                      const __global FLOAT_TYPE         *all_leaf_nodes_ALPHA_set,
                      const  __global FLOAT_TYPE         *all_leaf_nodes_BETA_set,
                      const   __global        int         *all_constrains_num_of_each_leaf_nodes,
                      const   __global        int         *all_leaf_nodes_offsets_in_all_ALPHA,
                                        int          leaf_node_num,
                        __global        int         *all_leaf_nodes_ancestor_nodes_ids,
                        __global        int         *leaf_nodes_start_pos_in_sorted_data_set,
                        __global        int         *pts_num_in_sorted_leaf_nodes,
                        __global FLOAT_TYPE         *dist_k_mins_global_tmp,
                        __global        int         *idx_k_mins_global_tmp,
                                        int          K_NN,
                        __global       long         *dist_computation_times_arr,
                        __global       long         *quadprog_times_arr,
                        __global       long         *dist_computation_times_in_quadprog)
{
    //---global thread id
    uint tid = get_global_id(0);
    //printf(" tid=%d, candidate_query_points_num=%d, ",tid, candidate_query_points_num);
    if(tid >= candidate_query_points_num){
        return;
    }

    uint j;


    //get the index of current query point
    int cur_query_point_index= candidate_query_points_indexes[tid];
    //printf("\n candidate_query_points_num=%d, cur_query_point_index=%d", candidate_query_points_num,cur_query_point_index);
    //---count the distance computation times in approximate quadprog.
    long cur_dist_compute_times_in_appr_quadprog=0;

    //---init the distance as MAX_FLOAT_TYPE
    for (int i=0;i<K_NN;i++){
        dist_k_mins_global_tmp[cur_query_point_index*K_NN+i]=MAX_FLOAT_TYPE;
    }

    int cur_query_points_appr_leaf_node_indexes= candidate_query_points_appr_leaf_node_indexes[tid];
    int cur_leaf_node_start_pos= leaf_nodes_start_pos_in_sorted_data_set[cur_query_points_appr_leaf_node_indexes];
    /*---------------------------------------------------------------------------------------------------------------
        //---query_points_nodes_alpha_scalar_product is not used in is_appr_min_dist_from_q_larger_by_hyper_plane now.
        //---because visiting  query_points_nodes_alpha_scalar_product randomly seems slow.
        //---private scalar product between current query point and  all ALPHAs, which are all initialized to 0.
        //---each node has a alpha constraint, a well as constraints of its ancestors nodes.
        //---'ALL_NODES_NUM' will be written before kernel is created.
    ----------------------------------------------------------------------------------------------------------------*/
        FLOAT_TYPE query_points_nodes_alpha_scalar_product[NODES_NUM];
        for (int i=0;i<NODES_NUM;i++){
            query_points_nodes_alpha_scalar_product[i]=0;
        }

    /*-----------------Copy global data as local data: visiting global data is relative slow in devices-----------------------------*/
        int quadprog_times_private=0;

        //a copy of current dist_k_mins_global_tmp
        FLOAT_TYPE dist_mins_private[MAX_NN_NUM];
        //a copy of current idx_k_mins_global_tmp
        int idx_mins_private[MAX_NN_NUM];

        for (j=0; j<K_NN; j++){
            dist_mins_private[j] = MAX_FLOAT_TYPE;
            idx_mins_private [j] = -1;
        }
        //---here is tid instead of cur_query_point_index, tid is the offset of current query point in candidate_query_points_set
        __global FLOAT_TYPE* cur_query_point =candidate_query_points_set+tid*DIM;

        FLOAT_TYPE cur_query_point_private[DIM];
        //printf("\n cur_query_point_private[%d]=[",cur_query_point_index);
        for (int i=0;i<DIM;i++){
            cur_query_point_private[i]=cur_query_point[i];
            //printf(" %f ,",cur_query_point_private[i]);
        }
        //printf("]\n ");
    /*-----------------------------------------------------------------------------------------------------------------------------*/

    long dist_computation_times_tmp=0;
    int pts_num= pts_num_in_sorted_leaf_nodes[cur_query_points_appr_leaf_node_indexes];
    //printf("\n approximate leafnode pts_num=%d \n", pts_num);
    //---find approximate kNN in its approximate nodes.
    do_brute_force_and_update_private( cur_query_point_private, cur_query_point_index,
                                       pts_num, cur_leaf_node_start_pos,
                                       all_sorted_data_set, sorted_data_set_indexes,
                                       dist_mins_private, idx_mins_private,
                                       K_NN);

    //---add distance computation times
    dist_computation_times_tmp+=pts_num;

    for (int i=0;i<leaf_node_num;i++) {
        if (i==cur_query_points_appr_leaf_node_indexes)
            continue;

        int alpha_offset  = all_leaf_nodes_offsets_in_all_ALPHA[i];
        int constrains_num= all_constrains_num_of_each_leaf_nodes[i];

        //---get the current k^th min_dist_square of current query point
        FLOAT_TYPE cur_k_min_dist_square=dist_mins_private[K_NN-1];

        __global FLOAT_TYPE* cur_ALPHAT= all_leaf_nodes_ALPHA_set + alpha_offset*DIM;
        __global FLOAT_TYPE* cur_BETA  = all_leaf_nodes_BETA_set  + alpha_offset;

        //---add approximate quadprog times
        quadprog_times_arr[cur_query_point_index]++;

        //---the number of ancestor nodes is the same as the size of constraints
        __global int* cur_ancestor_nodes_ids= all_leaf_nodes_ancestor_nodes_ids + alpha_offset;

        //---check whether the current node is a candidate for current query point
        if (!is_appr_min_dist_from_q_larger_by_hyper_plane(cur_query_point_private,cur_ALPHAT,cur_BETA,
                                                           constrains_num,cur_k_min_dist_square,
                                                           &cur_dist_compute_times_in_appr_quadprog,
                                                           query_points_nodes_alpha_scalar_product,
                                                           cur_ancestor_nodes_ids ))
        {
            //---do brute force distance computation here, and update dist_k_mins_global_tmp and idx_k_mins_global_tmp
            //---get the number of points saved in current node
            //---i is cur leaf node index, not leaf node ori_index
            int pts_num= pts_num_in_sorted_leaf_nodes[i];
            int cur_leaf_node_start_pos= leaf_nodes_start_pos_in_sorted_data_set[i];

            do_brute_force_and_update_private( cur_query_point_private, cur_query_point_index,
                                               pts_num, cur_leaf_node_start_pos,
                                               all_sorted_data_set, sorted_data_set_indexes,
                                               dist_mins_private, idx_mins_private,
                                               K_NN);
            //---add distance computation times
            dist_computation_times_tmp+=pts_num;
        }
    }

    //---write distances and indices to global buffers (coalesced access)
    for (j=0; j<K_NN; j++){
        dist_k_mins_global_tmp[cur_query_point_index*K_NN+j] = dist_mins_private[j];
        idx_k_mins_global_tmp [cur_query_point_index*K_NN+j] = idx_mins_private[j];
    }

    //---save the dist computation times
    dist_computation_times_arr[cur_query_point_index]= dist_computation_times_tmp ;

    //---save the dist computation times in 'is_appr_min_dist_from_q_larger_by_hyper_plane'
    dist_computation_times_in_quadprog[cur_query_point_index]=cur_dist_compute_times_in_appr_quadprog;
}



/*
    //find kNN by brute force
    0 data_set                          : the number of current candidate query points
    1 data_set_size                     : cardinal
    2 query_points                      : all query points
    3 query_points_size                 : the length of query_points
    4 dist_k_mins_global_tmp            : the K min-distance of all query points,
                                          the length of dist_mins_global_tmp is K* query_points_size
    5 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
                                          the length of dist_mins_global_tmp is K* query_points_size
    6 K_NN                              : the value of K
*/
__kernel void do_brute_force_KNN(
                                __global  FLOAT_TYPE *data_set,
                                int       data_set_size,
                                __global  FLOAT_TYPE *query_points,
                                int       query_points_size,
                                __global  FLOAT_TYPE *dist_k_mins_global_tmp,
                                __global  int *idx_k_mins_global_tmp,
                                int       K_NN
                                )
{
    // global thread id
    uint tid = get_global_id(0);
    //printf("tid =%d \n",tid);

    if(tid >= query_points_size){
        return;
    }
    //printf("tid=%d, data_set_size =%d,query_points_size=%d  \n",tid,data_set_size,query_points_size);


    uint current_query_point_index=tid;

    //---init the distance as MAX_FLOAT_TYPE
    for (int i=0;i<K_NN;i++){
        dist_k_mins_global_tmp[current_query_point_index*K_NN+i]=MAX_FLOAT_TYPE;
    }

    //get the current k^th min_dist_square of current query point
    FLOAT_TYPE cur_k_min_dist_square=dist_k_mins_global_tmp[current_query_point_index*K_NN+K_NN-1];
    //printf("cur_k_min_dist_square =%f \n",cur_k_min_dist_square);


    FLOAT_TYPE dist_square_tmp=0;
    FLOAT_TYPE tmp=0;
    int tmp_idx=0;

    //local copy
    FLOAT_TYPE cur_query_point_private[DIM];
    for (int i=0;i<DIM;i++){
        cur_query_point_private[i]=query_points[current_query_point_index*DIM+i];
    }


    for (int i=0;i<data_set_size;i++){
        dist_square_tmp=0;
        cur_k_min_dist_square= dist_k_mins_global_tmp[current_query_point_index*K_NN+K_NN-1];
        for (int j=0;j<DIM;j++){
            tmp= data_set[i*DIM+j]-cur_query_point_private[j];
            dist_square_tmp+=tmp*tmp;
            //printf("tmp =%f, dist_square_tmp=%f\n",tmp,dist_square_tmp);
        }
        //printf("dist_square_tmp =%f, cur_k_min_dist_square=%f \n",dist_square_tmp, cur_k_min_dist_square);
        if (cur_k_min_dist_square> dist_square_tmp){
            //printf("update dist_k_mins_global_tmp...\n");
            int j = K_NN - 1;
            dist_k_mins_global_tmp[current_query_point_index*K_NN+j] = dist_square_tmp;
            int pts_idx =i;
            idx_k_mins_global_tmp[current_query_point_index*K_NN+j] = pts_idx;

            for(;j>0;j--){
                if(dist_k_mins_global_tmp[current_query_point_index*K_NN+j-1] > dist_k_mins_global_tmp[current_query_point_index*K_NN+j]){
                    //printf("new nn found, swap...");
                    tmp=dist_k_mins_global_tmp[current_query_point_index*K_NN+j-1];
                    dist_k_mins_global_tmp[current_query_point_index*K_NN+j-1]=dist_k_mins_global_tmp[current_query_point_index*K_NN+j];
                    dist_k_mins_global_tmp[current_query_point_index*K_NN+j]=tmp;

                    //swap indices
                    tmp_idx=idx_k_mins_global_tmp[current_query_point_index*K_NN+j-1];
                    idx_k_mins_global_tmp[current_query_point_index*K_NN+j-1]=idx_k_mins_global_tmp[current_query_point_index*K_NN+j];
                    idx_k_mins_global_tmp[current_query_point_index*K_NN+j]=tmp_idx;
                } else break;
            }
        }
    }
}


    int get_min_k_index(FLOAT_TYPE  *dist_k_mins_private_tmp, int K_NN){
        FLOAT_TYPE tmp_max=dist_k_mins_private_tmp[0];
        int result=0;
        for (int i=1;i<K_NN;i++){
            if (dist_k_mins_private_tmp[i]>tmp_max){
                result=i;
                tmp_max=dist_k_mins_private_tmp[i];
            }
        }
        return result;
    }

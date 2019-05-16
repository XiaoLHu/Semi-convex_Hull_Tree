/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */
#ifndef GPU_Circle_Queue_Convex_Tree_H
#define GPU_Circle_Queue_Convex_Tree_H
#include "cyw_types.h"
template <class T>
class GPU_Circle_Queue_Convex_Tree :public Convex_Tree<T>
{
public:
    GPU_Circle_Queue_Convex_Tree();
    virtual ~GPU_Circle_Queue_Convex_Tree(){
        delete [] this->dist_square_k_mins_global;
        delete [] this->idx_k_mins_global;
        delete [] this->dist_computation_times_arr_in_device;
        delete [] this->quadprog_times_arr_in_device;
    };
    int MAX_NN_NUM=100;

    //leaf_pts_percent : the percent that specify the number of leaf_pts_num
    GPU_Circle_Queue_Convex_Tree(Matrix<T>& data, FLOAT_TYPE leaf_pts_percent, int alg_type);

    //after kNN processing, get the result by the following three entries
    //they are all virtual procedures, and override here.
    virtual FLOAT_TYPE*  get_kNN_dists(int query_point_index){
        //return this->dist_
    };

    virtual FLOAT_TYPE*  get_kNN_dists_squre(int query_point_index){
        return this->dist_square_k_mins_global+ query_point_index*(this->K);
    };

    virtual  int* get_kNN_indexes(int query_point_index){
        return this->idx_k_mins_global + query_point_index* (this->K);
    };

    virtual void print_kNN_running_time_info();

    virtual void set_batch_size(int batch_size);

protected:
    //dist_k_mins_global and idx_k_mins_global will be initialize in 'kNN_XXX()'
    FLOAT_TYPE* dist_square_k_mins_global=NULL;
    int  *idx_k_mins_global=NULL;

    //dist_computation_times_arr_in_device[i] saves total distance computation times of the i^th query point,
    //and quadprog_times_arr_in_device[i] approximate quadprog times of the i^th query point.
    cl_long *dist_computation_times_arr_in_device=NULL;
    cl_long *dist_computation_times_arr_in_quadprog=NULL;

    //sum of all distance computation times saved in computation_statistic_num[0,2,4....]
    long long total_dist_computation_times=0;
    //sum of all quadprog times saved in computation_statistic_num[1,3,5....]
    long total_quadprog_times=0;

    //override do_kNN defined in super class, and implement specific algorithm here.
    virtual void do_kNN(Matrix<T> &query_points_mat);

    //for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
    void do_find_approximate_nodes();

    /*
        This is virtual procedure, a entry for specific task in subclass.
        it will be call in init_process_query_points defined in  superclass 'Convex_tree'
    */
    void virtual init_kNN_result();

    //override
    void virtual do_print_kNN_result(){

        int num_query_pts= this->query_points_vec.size();
        for (int i=0;i<num_query_pts;i++){
            std::cout<<i <<" ";
            std::cout<<"\n   the dist_squre of kNN of "<<i<<"{th} query point:";
            for (int j=0;j<this->K;j++){
                std::cout<<this->dist_square_k_mins_global[i*this->K+j]<<", ";
            }
            std::cout<<"\n   the indexes of kNN of "<<i<<"{th} query point:";
            for (int j=0;j<this->K;j++){
                std::cout<<this->idx_k_mins_global[i*this->K+j]<<", ";
            }
        }
    }
private:
    int BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE;
    /* it is not good to declare the following 2 vars here, please refactoring it next time */
    //it is used in find_a_leaf_node_for_query_point to figure out whether the leaf node is found.
    bool is_leaf_node_found;
    //it is also used in find_a_leaf_node_for_query_point to figure out the leaf node index which is found.
    int cur_leaf_node_found=-1;


    //if current_visiting_node_indexes_of_query_points[i]=j, then it means the i^th query point currently visiting the j^th node
    //during find kNN by traversing the tree.
    std::vector<int> current_visiting_node_indexes_of_query_points;

    openCL_stru os;

    //init devices for openCL
    void init_openCL_device();

    //init parameters 3-14 for train data set.
    void init_openCL_sorted_data_set();

    //init the running time parameters which include the approximate k_min_dists_square and k_min_indexes
    void init_final_NN_result_opencl_params( Matrix<T>& query_points_mat, int K);

    //init parameters 0-2, and 17 for train data set, here just allocate space for them without writing data
    void init_dynamic_batch_query_points();

    //here write the batch query data to device
    void recreate_dynamic_batch_query_data_param(FLOAT_TYPE *query_points,
                                                        int *query_point_indexes,
                                                        int  query_points_num,
                                                        int *query_point_approximate_leaf_nodes);
    //here write the batch query data to device
    void write_dynamic_batch_query_data_param(FLOAT_TYPE *query_points,
                                                     int *query_point_indexes,
                                                     int  query_points_num,
                                                     int *query_point_approximate_leaf_nodes);

    //current_visiting_node_indexes_of_query_points[i]=j means that the j^th node is the current
    //visiting node of the i^th query point.
    void init_current_visiting_nodes_for_all_query_points();

    void init_global_final_NN_result(int query_points_num, int K);


};


template <class T>
GPU_Circle_Queue_Convex_Tree<T>::GPU_Circle_Queue_Convex_Tree(Matrix<T>& data,
                                    FLOAT_TYPE leaf_pts_percent,
                                    int alg_type):Convex_Tree<T>(data, leaf_pts_percent,alg_type)
{
    //init platforms and devices
    init_openCL_device();
    //init parameters 3-14 for train data set.
    init_openCL_sorted_data_set();

    //init parameters 0-2, and 17 for train data set, here just allocate space for them without writing data
    init_dynamic_batch_query_points();
}


//for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::do_find_approximate_nodes(){
    int query_points_num= this->query_points_vec.size();
    this->timer_total_approximate_searching.start_my_timer();

    this->appr_leaf_node_indexes.resize(query_points_num);
    //for each query point, we find its approximate nodes, and save it in appr_leaf_node_indexes.
    for (int i=0;i<query_points_num;i++){
        int cur_node_index=0;
        Vector<T>& q= this->query_points_vec[i];
        while (this->nodes[cur_node_index]->isLeaf==false){
            //this->nodes[cur_node_index]->print()  ;
            Vector<T>& left_center_vec=this->nodes[cur_node_index]->left_center;
            Vector<T>& right_center_vec=this->nodes[cur_node_index]->right_center;
            FLOAT_TYPE left_dist_squre=pdist2_squre(q,left_center_vec);
            FLOAT_TYPE right_dist_squre=pdist2_squre(q,right_center_vec);
            //count the distance computation times.
            this->dist_computation_times_in_host+=2;
            if (left_dist_squre>=right_dist_squre){
                cur_node_index=this->nodes[cur_node_index]->right;
            } else
                cur_node_index=this->nodes[cur_node_index]->left;
        };
        this->appr_leaf_node_indexes[i]=this->nodes[cur_node_index]->leaf_index;
    }
    this->timer_total_approximate_searching.stop_my_timer();
}

//override
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>:: set_batch_size(int batch_size){
    cl_int status;
    this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE=batch_size;
    status = clReleaseMemObject(os.candidate_query_points_indexes_buffer);                  //parm  1
    status = clReleaseMemObject(os.candidate_query_points_set_buffer);                      //parm  2
    status = clReleaseMemObject(os.candidate_approximate_leaf_nodes_buffer);                //param 3
    this->init_dynamic_batch_query_points();
}

//override do_kNN defined in super class, and implement specific algorithm here.
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::do_kNN(Matrix<T> &query_points_mat){
    /*
        OpenCL parameters 4-16 are already initialized in 'init_openCL_device() and init_openCL_sorted_data_set()'
        called in ctor, and 'init_final_NN_result_opencl_params()' called in init_other_task();
        the remains parameters are 0-3, they should be prepared here and pushed to device to perform kNN task.
    */
    //---prepare data for openCL parameters 0-3.
    //---compute how many batches
    int total_query_points_num=this->query_points_vec.size();
    this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE=this->m_data.nrows();
    int batch_num= ceil(total_query_points_num/this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE);

    /*------------------------------------------PROCESS EACH BATCH----------------------------------------------------*/
    T   *query_points_raw_data= query_points_mat.get_matrix_raw_data();
    for (int i=0;i<batch_num;i++){
    }
    /*------------------------------------------PROCESS EACH BATCH----------------------------------------------------*/


    //read the final result from devices and save into this->dist_square_k_mins_global and  this->idx_k_mins_global
    status = clEnqueueReadBuffer(os.commandQueue, this->os.dist_k_mins_global_buffer , CL_TRUE, 0,
                                 total_query_points_num*this->K* sizeof(FLOAT_TYPE),this->dist_square_k_mins_global , 0, NULL, NULL);
    status = clEnqueueReadBuffer(os.commandQueue, this->os.idx_k_mins_global_buffer , CL_TRUE, 0,
                                 total_query_points_num*this->K* sizeof(int), this->idx_k_mins_global, 0, NULL, NULL);

    status = clEnqueueReadBuffer(os.commandQueue, this->os.dist_computation_times_arr_in_device_buffer , CL_TRUE, 0,
                                 total_query_points_num* sizeof(cl_long), this->dist_computation_times_arr_in_device, 0, NULL, NULL);
    status = clEnqueueReadBuffer(os.commandQueue, this->os.dist_computation_times_arr_in_quadprog_buffer , CL_TRUE, 0,
                                 total_query_points_num* sizeof(cl_long), this->quadprog_times_arr_in_device, 0, NULL, NULL);
    //do statistic
    for (int i=0; i<total_query_points_num;i++){
        this->total_dist_computation_times+=this->dist_computation_times_arr_in_device[i];
        this->total_quadprog_times        +=this->quadprog_times_arr_in_device[i];
        //std::cout<<"\n dist_computation_times_arr_in_device="<<this->dist_computation_times_arr_in_device[i];
        //std::cout<<"\n quadprog_times_arr_in_device="<<this->quadprog_times_arr_in_device[i];
    }

    delete[] cur_batch_indexes;
    delete[] cur_batch_candidate_leaf_nodes_indexes;

}


template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_openCL_device(){
    cl_int err,status;
    /*Step 1: Getting platforms and choose an available one(first).*/
    getPlatform_ids(os.platform_ids,os.num_platforms);
    os.platform=os.platform_ids[0];
    /*Step 2:Query the platform and choose the first GPU device if has one.*/
    os.devices=getCl_device_id(os.platform);

    /*Step 3: Create context.*/
    os.context = clCreateContext(NULL,1, os.devices,NULL,NULL,NULL);

    /*Step 4: Creating command queue associate with the context.*/
    os.commandQueue = clCreateCommandQueue(os.context, os.devices[0], 0, NULL);


    /*Step 5: Create program object */
    // constants for all kernels
    int dim= this->m_data.ncols();
    int all_contrains_num=0;

    for (int i=0;i<this->nodes_num;i++){
        int tmp_num= this->nodes[i]->BETA.size();
        all_contrains_num+=tmp_num;
    }

	char constants[MAX_KERNEL_CONSTANTS_LENGTH];
	sprintf(constants,
			"#define DIM %d\n\
			 #define NODES_NUM %d\n\
			 #define ALL_CONSTRAINS_NUM %d\n\
			 #define MAX_NN_NUM %d\n\
			 #define USE_DOUBLE %d\n",
			dim, this->nodes_num, all_contrains_num,this->MAX_NN_NUM,USE_DOUBLE);
    os.kernel_filename_brute_computing_distances = "kernels/do_kNN_gpu.cl";

    os.do_kNN_kernel=make_kernel_from_file(os,constants,  "do_finding_KNN_by_circle_queue");
}


/*
    0 candidate_query_queue_len         : the number of query queue
    1 candidate_query_points_indexes    : the indexes of current query points in all query points set
    2 candidate_query_points_set        : the current query points data set
    3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
 */
//init parameters 1-3 for train data set, here just allocate space for them without writing data
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_dynamic_batch_query_points(){
    //in order to align m_data, we have to make the query size equal to the length of m_data.
    int query_points_num=this->m_data.nrows();
    int dim=this->m_data.ncols();
    cl_int err, status;
    //param 1
    os.candidate_query_points_indexes_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY ,
                                                              query_points_num * sizeof(int),
                                                              NULL, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 2
    os.candidate_query_points_set_buffer     = clCreateBuffer(os.context, CL_MEM_READ_ONLY ,
                                                              query_points_num *dim* sizeof(FLOAT_TYPE),
                                                              NULL, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 3
    os.candidate_approximate_leaf_nodes_buffer= clCreateBuffer(os.context, CL_MEM_READ_ONLY ,
                                                               query_points_num * sizeof(int),
                                                               NULL, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //the sorted_data and the dists_squre can be pushed to devices before executing kNN processing
/*
    0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                          is too large, we can submit subset of query sets to this kernel.
    1 candidate_query_points_indexes    : the indexes of current query points in all query points set
    2 candidate_query_points_set        : the current query points data set
    3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
    ....
*/
    status |= clSetKernelArg(os.do_kNN_kernel, 1,  sizeof(cl_mem), &os.candidate_query_points_indexes_buffer );
    status |= clSetKernelArg(os.do_kNN_kernel, 2,  sizeof(cl_mem), &os.candidate_query_points_set_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 3,  sizeof(cl_mem), &os.candidate_approximate_leaf_nodes_buffer);
}

//init parameters 3-14 for train data set.
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_openCL_sorted_data_set(){
    int dim= this->m_data.ncols();

    int data_len = this->getSortedData().nrows();

    cl_int err, status;
    //param 4 :all_sorted_data_set
    T* sorted_data=this->getSortedData().get_matrix_raw_data();
    os.sorted_data_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           data_len *dim* sizeof(T), (void *)sorted_data, &err);
	check_cl_error(err, __FILE__, __LINE__);

	//param 5: sorted_data_set_indexes
	int* sorted_data_indexes= this->getSortedData_ori_indexes().get_matrix_raw_data();
	os.sorted_data_indexes_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   data_len * sizeof(int),(void *)sorted_data_indexes, &err);
	check_cl_error(err, __FILE__, __LINE__);

    //param 5: simplified_tree_stru_buffer
    CONVEX_TREE* simplified_tree_stru=this->simplified_tree.get_matrix_raw_data() ;
    os.simplified_tree_stru_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    this->nodes_num * sizeof(CONVEX_TREE),
                                                    (void *)simplified_tree_stru, &err);
	check_cl_error(err, __FILE__, __LINE__);


    //param 6: all_ALPHA_set_buffer
    int all_alpha_num= 0;
    for (int i=0;i<this->nodes_num;i++){
        all_alpha_num+= this->nodes[i]->ALPHA.nrows();
    }
    FLOAT_TYPE* all_ALPHA_set= new FLOAT_TYPE[all_alpha_num*dim];

    int cur_pos=0;

    //the first node is root who has no alpha and beta
    for (int i=1;i<this->nodes_num;i++){
        int tmp_num= this->nodes[i]->ALPHA.nrows();
        //copy data
        memcpy(all_ALPHA_set+cur_pos*dim, this->nodes[i]->ALPHA.get_matrix_raw_data(),tmp_num*dim*sizeof(FLOAT_TYPE));
        cur_pos+=tmp_num;
    }
    os.all_ALPHA_set_buffer= clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            all_alpha_num*dim * sizeof(FLOAT_TYPE),
                                            (void *)all_ALPHA_set, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 7: all_BETA_set_buffer
    FLOAT_TYPE* all_BETA_set= new FLOAT_TYPE[all_alpha_num];
    cur_pos=0;
    //the first node is root who has no alpha and beta
    for (int i=1;i<this->nodes_num;i++){
        int tmp_num= this->nodes[i]->BETA.size();
        //copy data
        memcpy(all_BETA_set+cur_pos, this->nodes[i]->BETA.get_matrix_raw_data(),tmp_num*sizeof(FLOAT_TYPE));
        cur_pos+=tmp_num;
    }
    os.all_BETA_set_buffer= clCreateBuffer( os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            all_alpha_num* sizeof(FLOAT_TYPE),
                                            (void *)all_BETA_set, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 8: all_constrains_num_of_each_nodes_buufer
    int* all_constrains_num_of_each_nodes= new int[this->nodes_num];
    cur_pos=0;
    for (int i=0;i<this->nodes_num;i++){
        int tmp_num= this->nodes[i]->BETA.size();
        all_constrains_num_of_each_nodes[i]=tmp_num;
    }
    os.all_constrains_num_of_each_nodes_buffer= clCreateBuffer( os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                this->nodes_num* sizeof(int),
                                                                (void *)all_constrains_num_of_each_nodes, &err);
    //param 9: all_nodes_offsets_in_all_ALPHA_buffer
    int* all_nodes_offsets_in_all_ALPHA= new int[this->nodes_num];
    cur_pos=0;
    for (int i=0;i<this->nodes_num;i++){
        int tmp_num= this->nodes[i]->ALPHA.nrows();
        all_nodes_offsets_in_all_ALPHA[i]=cur_pos;
        cur_pos+=tmp_num;
    }
    os.all_nodes_offsets_in_all_ALPHA_buffer= clCreateBuffer( os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                              this->nodes_num* sizeof(int),
                                                              (void *)all_nodes_offsets_in_all_ALPHA, &err);
    check_cl_error(err, __FILE__, __LINE__);


    //param 10:
    int* sorted_leaf_nodes_start_pos_in_sorted_data_set= new int[leaf_nodes_num];
    for (int i=0;i<leaf_nodes_num;i++){
        sorted_leaf_nodes_start_pos_in_sorted_data_set[i]=this->leaf_nodes_start_pos_in_data_set[i];
    }

    os.sorted_leaf_nodes_start_pos_in_sorted_data_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                          leaf_nodes_num * sizeof(int),
                                                                          (void *)sorted_leaf_nodes_start_pos_in_sorted_data_set, &err);
	check_cl_error(err, __FILE__, __LINE__);

	//param 11
    int* pts_num_in_sorted_leaf_nodes= new int[leaf_nodes_num];
    for (int i=0;i<leaf_nodes_num;i++){
        int leaf_node_index= this->leaf_nodes_ori_indexes[i];
        pts_num_in_sorted_leaf_nodes[i]=this->nodes[leaf_node_index]->pts_number;
    }
	os.pts_num_in_sorted_leaf_nodes_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                            leaf_nodes_num * sizeof(int),
                                                            (void *)pts_num_in_sorted_leaf_nodes, &err);
	check_cl_error(err, __FILE__, __LINE__);



    //the sorted_data and the dists_squre can be pushed to devices before executing kNN processing
    /*
        0 candidate_query_queue_len         : the number of query queue
        1 candidate_query_points_indexes    : the indexes of current query points in all query points set
        2 candidate_query_points_set        : the current query points data set
        3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
        4 all_sorted_data_set               : all sorted data
        5 sorted_data_set_indexes           : all points indexes in sorted data set
        6 all_ALPHA_set                     : ALPHA set of all nodes
        7 all_BETA_set                      : BETA set of all nodes
        8 all_constrains_num_of_each_nodes  : all_constrains_num_of_each_nodes[i]=j means i^th nodes has j constrains, i.e. has j alphas and betas
        9 all_nodes_offsets_in_all_ALPHA    : the offset of each node in ALPHA
       10 sorted_leaf_nodes_start_pos_in_sorted_data_set : specify the start position from which each sorted leave node in sorted data set
       11 pts_num_in_sorted_leaf_nodes      : the length of points saved in each sorted leave node
       12 dist_k_mins_global_tmp            : the K min-distance of all query points,
                                              the length of dist_mins_global_tmp is K* query_points_size
       13 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
                                              the length of dist_mins_global_tmp is K* query_points_size
       14 K_NN                              : the value of K
       15 dist_computation_times_arr        : dist_computation_times_arr[i] saves total distance computation times of the i^th point.
       16 dist_computation_times_in_appr_quadprog : dist_computation_times_in_appr_quadprog[i] saves total distance computation
                                                    times of the i^th point in approximate quadprog.
    */
    status |= clSetKernelArg(os.do_kNN_kernel,  4, sizeof(cl_mem),  &os.sorted_data_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel,  5, sizeof(cl_mem),  &os.sorted_data_indexes_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel,  6, sizeof(cl_mem),  &os.all_ALPHA_set_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel,  7, sizeof(cl_mem),  &os.all_BETA_set_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel,  8, sizeof(cl_mem),  &os.all_constrains_num_of_each_nodes_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel,  9, sizeof(cl_mem),  &os.all_nodes_offsets_in_all_ALPHA_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 10, sizeof(cl_mem),  &os.sorted_leaf_nodes_start_pos_in_sorted_data_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 11, sizeof(cl_mem),  &os.pts_num_in_sorted_leaf_nodes_buffer);

    delete [] all_ALPHA_set;                                   //temp param 6
    delete [] all_BETA_set;                                    //temp param 7
    delete [] all_constrains_num_of_each_nodes;                //temp param 8
    delete [] all_nodes_offsets_in_all_ALPHA;                  //temp param 9
    delete [] sorted_leaf_nodes_start_pos_in_sorted_data_set;  //temp param 10
    delete [] pts_num_in_sorted_leaf_nodes;                    //temp param 11
}


//init running time parameters
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_final_NN_result_opencl_params( Matrix<T>& query_points_mat, int K){
    cl_int status,err;
    int dim= this->m_data.ncols();
    int query_points_num=query_points_mat.nrows();
	//param 15
    os.dist_k_mins_global_buffer= clCreateBuffer(os.context, CL_MEM_READ_WRITE ,
                                                 query_points_num*this->K* sizeof(FLOAT_TYPE),
                                                 NULL, &err);
    check_cl_error(err, __FILE__, __LINE__);


	check_cl_error(err, __FILE__, __LINE__);

	//param 16
    os.idx_k_mins_global_buffer= clCreateBuffer(os.context, CL_MEM_READ_WRITE,
                                                query_points_num*(this->K)* sizeof(int),
                                                NULL,
                                                &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 18
    this->dist_computation_times_arr_in_device= new cl_long[query_points_num];
    memset(this->dist_computation_times_arr_in_device,0, query_points_num*sizeof(cl_long));

	os.dist_computation_times_arr_in_device_buffer = clCreateBuffer(os.context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
                                                                    query_points_num* sizeof(cl_long),
                                                                    (void *)this->dist_computation_times_arr_in_device, &err);

    //param 19
    this->quadprog_times_arr_in_device= new cl_long[query_points_num];
    memset(this->quadprog_times_arr_in_device,0, query_points_num*sizeof(cl_long));

	os.quadprog_times_arr_in_device_buffer = clCreateBuffer(os.context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR,
                                                            query_points_num* sizeof(cl_long),
                                                            (void *)this->quadprog_times_arr_in_device, &err);
	check_cl_error(err, __FILE__, __LINE__);


    //the sorted_data and the dists_squre can be pushed to devices before executing kNN processing
    /* ....
       12 dist_k_mins_global_tmp            : the K min-distance of all query points,
                                              the length of dist_mins_global_tmp is K* query_points_size
       13 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
                                              the length of dist_mins_global_tmp is K* query_points_size
       14 K_NN                              : the value of K
       15 dist_computation_times_arr        : dist_computation_times_arr[i] saves total distance computation times of the i^th point.
       16 dist_computation_times_in_appr_quadprog : dist_computation_times_in_appr_quadprog[i] saves total distance computation
                                                    times of the i^th point in approximate quadprog.
    */
    status |= clSetKernelArg(os.do_kNN_kernel, 12, sizeof(cl_mem), &os.dist_k_mins_global_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 13, sizeof(cl_mem), &os.idx_k_mins_global_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 14, sizeof(int),    &K);
    status |= clSetKernelArg(os.do_kNN_kernel, 15, sizeof(cl_mem), &os.dist_computation_times_arr_in_device_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 16, sizeof(cl_mem), &os.dist_computation_times_arr_in_quadprog_buffer);
}

//here write the batch query data to device
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::recreate_dynamic_batch_query_data_param( FLOAT_TYPE* query_points,
                                                                         int* query_point_indexes,
                                                                         int  query_points_num,
                                                                         int* query_point_approximate_leaf_nodes)
{
    cl_int status,err;
    int dim= this->m_data.ncols();
    if (this->cur_batch_iter>0){
        status = clReleaseMemObject(os.candidate_query_points_indexes_buffer);                //parm  1
        status|= clReleaseMemObject(os.candidate_query_points_set_buffer);
        status|= clReleaseMemObject(os.candidate_approximate_leaf_nodes_buffer);
    }
    os.candidate_query_points_indexes_buffer = clCreateBuffer(os.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                              query_points_num * sizeof(int),
                                                              query_point_indexes, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 2
    os.candidate_query_points_set_buffer     = clCreateBuffer(os.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR ,
                                                              query_points_num *dim* sizeof(FLOAT_TYPE),
                                                              query_points, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //param 17
    os.candidate_approximate_leaf_nodes_buffer= clCreateBuffer(os.context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR ,
                                                               query_points_num * sizeof(int),
                                                               query_point_approximate_leaf_nodes, &err);
    check_cl_error(err, __FILE__, __LINE__);

    //the sorted_data and the dists_squre can be pushed to devices before executing kNN processing
    /*  the parameters: here we only initialize those final parameters: 0-2, 15-17
        0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                              is too large, we can submit subset of query sets to this kernel.
        1 candidate_query_points_indexes    : the indexes of current query points in all query points set
        2 candidate_query_points_set        : the current query points data set
        .....
    */
    status |= clSetKernelArg(os.do_kNN_kernel, 0,  sizeof(int),    &query_points_num);
    status |= clSetKernelArg(os.do_kNN_kernel, 1,  sizeof(cl_mem), &os.candidate_query_points_indexes_buffer );
    status |= clSetKernelArg(os.do_kNN_kernel, 2,  sizeof(cl_mem), &os.candidate_query_points_set_buffer);
    status |= clSetKernelArg(os.do_kNN_kernel, 3,  sizeof(cl_mem) , &os.candidate_approximate_leaf_nodes_buffer);
}

//here write the batch query data to device
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::write_dynamic_batch_query_data_param(FLOAT_TYPE* query_points,
                                                                                  int* query_point_indexes,
                                                                                  int  query_points_num,
                                                                                  int* query_point_approximate_leaf_nodes)
{
    cl_int status,err;
    int dim= this->m_data.ncols();

    //param 1: write data to buffers
    err = clEnqueueWriteBuffer(os.commandQueue,
                               os.candidate_query_points_indexes_buffer,
                               CL_FALSE, 0,
                               query_points_num * sizeof(int),
                               query_point_indexes, 0, NULL, NULL);
    check_cl_error(err, __FILE__, __LINE__);

    //param 2
    err = clEnqueueWriteBuffer(os.commandQueue,
                               os.candidate_query_points_set_buffer,
                               CL_FALSE, 0,
                               query_points_num *dim* sizeof(FLOAT_TYPE),
                               query_points, 0, NULL, NULL);
    check_cl_error(err, __FILE__, __LINE__);


    //param 17
    err = clEnqueueWriteBuffer(os.commandQueue,
                               os.candidate_approximate_leaf_nodes_buffer,
                               CL_FALSE, 0,
                               query_points_num * sizeof(int),
                               query_point_approximate_leaf_nodes, 0, NULL, NULL);

    check_cl_error(err, __FILE__, __LINE__);
    status |= clSetKernelArg(os.do_kNN_kernel, 0,  sizeof(int),    &query_points_num);
}

/*
    This is virtual procedure, a entry for specific task in subclass.
    it will be call in init_process_query_points defined in  superclass 'Convex_tree'
*/
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_kNN_result()
{
    //init final result
    int query_points_num=this->query_points_vec.size();
    //init dist_square_k_mins_global and idx_k_mins_global
    init_global_final_NN_result(query_points_num, this->K);

    //for all
    this->do_find_approximate_nodes();

    //init  opencl parameters for kNN result
    init_final_NN_result_opencl_params(this->m_data, this->K);
}

template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::init_global_final_NN_result(int query_points_num, int K){
    if (this->dist_square_k_mins_global!=NULL){
        delete []this->dist_square_k_mins_global;
        delete []this->idx_k_mins_global;
    }
    this->dist_square_k_mins_global= new FLOAT_TYPE[query_points_num*K];

    this->idx_k_mins_global= new int[query_points_num*K];
    memset(this->idx_k_mins_global,-1,sizeof(int)*query_points_num*K);
}

//virtual procedure
template <typename T>
void GPU_Circle_Queue_Convex_Tree<T>::print_kNN_running_time_info()
{

    std::cout<<"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n";
    std::cout<<"*     Query points number:"<<this->query_points_vec.size()<<", and K="<<this->K<<"\n";
    std::cout<<"*     GPU Alorithm ";
    this->timer_whole_alg.print();
    std::cout<<"*          Where:\n";
    this->timer_init_process_query_points.print("*          (1) Init query points: ");
    this->timer_total_approximate_searching.print("*              where Approximate searching: ");
    std::cout<<"*          (2) Quadratic programming number in devices="<< this->total_quadprog_times<<" \n";

    std::cout<<"*          (3) Distance computations number in devices="<< this->total_dist_computation_times<<" \n";

    std::cout<<"*          (4) Distance computations in approximate searching in host="<<this->dist_computation_times_in_host<<" \n";
    std::cout<<"*          (5) Overall distance computations in host and devices=(3)+(4)="<<this->total_dist_computation_times+this->dist_computation_times_in_host<<" \n";
    std::cout<<"*          (6) BATCH_SIZE="<< this->BATCH_SIZE_OF_QUERY_POINTS_PUSCH_TO_DEVICE <<", batch iteration times="<<this->cur_batch_iter<<"\n";
    std::cout<<"*-----------------------------------------KNN QUERY RESULT----------------------------------------";

};




#endif // GPU_Circle_Queue_Convex_Tree_H

/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */
 //#include <time.h>
#include <CL/cl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>
#include "basic_functions.h"
#include "ConvexNode.h"
#include "Array.hh"
#include "Convex_Tree.h"
#include "openCL_tool.h"
#include "CPU_Convex_Tree.h"
#include "GPU_Convex_Tree.h"
#include "cyw_types.h"
#include "data_processor.h"
//gof. design patterns: class helper

int NOISE_RATE=100;
class Tree_Helper{
public:
     //alg_type specifies the this->ALG_TYPE
    template<typename T>
    static Convex_Tree<T>* create_a_tree(int alg_type, Matrix<T>& data, FLOAT_TYPE leaf_pts_percent){
        Convex_Tree<T>* result=NULL;
        if (alg_type==USE_CPU_RECURSIVE_APPRXIMATE_QP|| alg_type==USE_CPU_LEAF_APPRXIMATE_QP ){
            result= new CPU_Convex_Tree<T>(data,leaf_pts_percent,alg_type);
        }

        if (alg_type==USE_GPU_RECURSIVE_APPRXIMATE_QP||alg_type==USE_GPU_LEAF_APPRXIMATE_QP)
            result= new GPU_Convex_Tree<T>(data, leaf_pts_percent,alg_type);

        if (alg_type==USE_GPU_CIRCLE_QUEUE_APPRXIMATE_QP)
            //result= new GPU_Circle_Queue_Convex_Tree<T>(data, leaf_pts_percent,alg_type);

        if (result!=NULL)
            result->set_alg_type(alg_type);
        return result;
    }
};

Vector<NNResult<FLOAT_TYPE>* > brute_force_check(Matrix<FLOAT_TYPE>& data,
                                                 Matrix<FLOAT_TYPE>& q_points,
                                                 int K,
                                                 Convex_Tree<FLOAT_TYPE>* cct )
{
    std::cout<<"\n brute force is starting...." ;
    int dim=data.ncols();
    long start_t=clock();//GetTickCount();
    Vector<NNResult<FLOAT_TYPE>* > results;
    results.resize(q_points.nrows());

    for (int i=0;i<q_points.nrows();i++){

        FLOAT_TYPE *q = q_points.get_matrix_raw_data()+i*dim;
        Vector<FLOAT_TYPE> tmp_dists_square=pdist2_squre(data, q,dim);
        results[i] = new NNResult<FLOAT_TYPE>();
        results[i]->init(K);
        for (int j=0; j<tmp_dists_square.size();j++){
            results[i]->update(tmp_dists_square[j],j);
        }

        bool err=false;
        for (int j=0;j<K;j++){
            FLOAT_TYPE cct_result=cct->get_kNN_dists_squre(i)[j];
            FLOAT_TYPE brute_result=results[i]->k_dists_squre[j];
            int cct_result_index=cct->get_kNN_indexes(i)[j];
            int brute_result_index=results[i]->k_indexes[j];
            if ( (abs(cct_result-brute_result)/cct_result>0.001)&&
                 (cct_result_index!=brute_result_index)){
                std::cout<<"\n error found, at point index="<<i<<" and j="<<j;
                err=true;
                break;
            }
        }
        if (err){
            FLOAT_TYPE* tmp=cct->get_kNN_dists_squre(i);
            int* tmp_indexes=cct->get_kNN_indexes(i);
            std::cout<<" \n knn dist squre of query point "<<i <<" is: ";
            for (int i=0;i<K;i++) {
                std::cout<<tmp[i]<<", ";
            }

            std::cout<<" \n knn indexes of query point "<<i <<" is: ";
            for (int i=0;i<K;i++) {
                std::cout<<tmp_indexes[i]<<", ";
            }
            results[i]->print();
        } else{
             std::cout<<"correct "<<i<<", ";
        }


        //std::cout<<"\n i="<<i<<" ";
        //results[i]->print();
    }
    long end_t=clock();//GetTickCount();
    std::cout<<"\n brute force finished, running time is "<< end_t-start_t<<"\ms" ;

}

void test_on_random(){
   int K=10;
    int dim=10;
    Matrix<FLOAT_TYPE> data;

    /*------------------------------create a random data set-----------------------------------------*/
    data= Matrix<FLOAT_TYPE>::rand_matrix(200000,dim);
    data/=1000;
    //data.print();
    /*------------------------------create a random data set-----------------------------------------*/

    /*------------------------------create query points set randomly---------------------------------*/
    Matrix<FLOAT_TYPE> q_points=Matrix<FLOAT_TYPE>::rand_matrix(10000,dim);
    q_points /=1000;
    /*------------------------------create query points set randomly---------------------------------*/

    //create a convex tree
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    cct->print_tree_info();

    cct->kNN(q_points,K);
    cct->print_kNN_running_time_info();
    //cct->print_kNN_reult();

    Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);

}

void test_on_real_data(int K, char *data_file_name, FLOAT_TYPE leaf_percent, FILE* log_file){
    int dim;


    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);

    //---create data set for query
    Matrix<FLOAT_TYPE> data(raw_data,data_size,dim);
    data/=1000;
    //---data.print();


    /*-------------------------------- Select Data Points From Dataset Randomly -------------------------------*/
        int test_num=20000;
        //char* idx_filename="exp_random_idx_2w(buffer kd tree).txt";
        //float* xTest_idx=read_data(idx_filename,&dim1,&n_size1);
        //FILE* ran_idx_file= fopen("exp_random_idx_2w(convex_tree).txt","a+");
        //write_file_log("\n",ran_idx_file);

        Vector<int> indx(test_num);
        for (int i=0;i<test_num;i++){
            //randomly get an index \in [0,n_size-1]
            int idx= rand();
            indx[i]=idx%data_size;
            //char c[10];
            //itoa(idx,c,10);
            //write_file_log(c,ran_idx_file);
            //write_file_log("\n",ran_idx_file);
        }
        //fclose(ran_idx_file);

        Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);
        for (int i=0; i< q_points.nrows();i++){
            for (int j=0; j<q_points.ncols();j++){
                float tmp=rand()/NOISE_RATE;
                q_points[i][j] +=tmp;
            }
        }

        //q_points[1282].print();
    /*------------------------------ Select Data Points From Dataset Randomly ---------------------------------*/


    //create a convex tree
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    cct->print_tree_info();


    write_file_log("\n\n",log_file);
    write_file_log(data_file_name,log_file);
    char str_tmp[10];
    sprintf(str_tmp,"%f",leaf_percent);
    write_file_log(",     percent threshold: ",log_file);
    write_file_log(str_tmp,log_file);
    cct->save_tree_info(log_file);

    cct->kNN(q_points,K);
    cct->print_kNN_running_time_info();
    cct->save_KNN_running_time_info(log_file);

    cct->print_kNN_reult();
    //fclose(log_file);
    //Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);
    delete cct;
}



void test_on_real_data_test(int K, char *data_file_name, FLOAT_TYPE leaf_percent, FILE* log_file){
    int dim;


    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);

    //data_size=40000;
    //---create data set for query
    Matrix<FLOAT_TYPE> data(raw_data,data_size,dim);
    data/=1000;
    //---data.print();


    /*-------------------------------- Select Data Points From Dataset Randomly -------------------------------*/
        int test_num=20000;
        //char* idx_filename="exp_random_idx_2w(buffer kd tree).txt";
        //float* xTest_idx=read_data(idx_filename,&dim1,&n_size1);
        //FILE* ran_idx_file= fopen("exp_random_idx_2w(convex_tree).txt","a+");
        //write_file_log("\n",ran_idx_file);

        Vector<int> indx(test_num);
        for (int i=0;i<test_num;i++){
            //randomly get an index \in [0,n_size-1]
            int idx= rand();
            indx[i]=idx%data_size;
            //char c[10];
            //itoa(idx,c,10);
            //write_file_log(c,ran_idx_file);
            //write_file_log("\n",ran_idx_file);
        }
        //fclose(ran_idx_file);

        Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);
        //q_points[1282].print();
    /*------------------------------ Select Data Points From Dataset Randomly ---------------------------------*/


    //create a convex tree
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    cct->print_tree_info();

    cct->kNN(q_points,K);
//    for (int i=test_num-10;i<test_num;i++){
//         std::cout<<"\n finished: "<<i;
//         Vector<int> idx_tmp(1);
//         idx_tmp[0]=i;
//         Matrix<FLOAT_TYPE> tmp_q=q_points.extractRows(idx_tmp);
//         cct->kNN(tmp_q,K);
//    }
    cct->print_kNN_running_time_info();
    //cct->print_kNN_reult();


    //fclose(log_file);
    //std::cout<<"\n checking....";
    //Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);
    delete cct;
}


void test_on_real_data_massive(int K, int test_num_each_batch, int batch_num, char *data_file_name, FLOAT_TYPE leaf_percent, FILE* log_file,int platform_id, int batch_size){
    int dim;


    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);

    //---create data set for query
    Matrix<FLOAT_TYPE> data(raw_data,data_size,dim);
    //data/=1000;
    int dim_lim=70;
    if (dim>dim_lim){
        Vector<int> idx(dim_lim);
        for(int i=0;i<dim_lim;i++)
            idx[i]=i;
        data= data.extractColumns(idx);
    }
    //---data.print();

    GPU_PLATFORM_ID=platform_id;
    //create a convex tree
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    cct->set_batch_size(batch_size);
    cct->print_tree_info();

    write_file_log("\n\n",log_file);
    write_file_log(data_file_name,log_file);
    char str_tmp[10];
    sprintf(str_tmp,"%f",leaf_percent);
    write_file_log(",     percent threshold: ",log_file);
    write_file_log(str_tmp,log_file);
    cct->save_tree_info(log_file);

     std::cout<<"\n queries generating ....";
    Vector<int> indx(test_num_each_batch*batch_num);
    for (int i=0;i<test_num_each_batch*batch_num;i++){
        int idx= rand();
        indx[i]=idx%data_size;
    }


    Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);

    for (int i=0; i< q_points.nrows();i++){
        for (int j=0; j<q_points.ncols();j++){
            float tmp=rand()/NOISE_RATE;
            q_points[i][j] +=tmp;
            //q_points[i][j] =tmp;
        }
    }

    std::cout<<"\n queries generation finished.\n";
    CYW_TIMER timer;
    timer.start_my_timer();
    cct->kNN(q_points,K);
    cct->print_kNN_running_time_info();
    cct->save_KNN_running_time_info(log_file);
    //Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);

    timer.stop_my_timer();


    char buffer[1000];
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------");
    strcat(buffer,"\n*     Query points number:");
    my_strcat(buffer,test_num_each_batch*batch_num);
    strcat(buffer,", and K=");
    my_strcat(buffer,K);
    strcat(buffer,"\n*     Noise= rand()/");
    my_strcat(buffer,NOISE_RATE);
    strcat(buffer,"\n*     GPU Alorithm ");
    timer.strcat_to_buffer(buffer);
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
    write_file_log(buffer,log_file);

    delete cct;
    //cct->print_kNN_reult();
    //fclose(log_file);

}


void test_on_real_data_Cardinal(int K, int test_num_each_batch, int batch_num, char *data_file_name, FLOAT_TYPE leaf_percent, FILE* log_file){
    int dim;


    int data_size;

    float *raw_data=read_data(data_file_name," ",&dim, &data_size);

    for (int i=1; i<batch_num+1;i++){
        //---create data set for query
        int tmp_len= test_num_each_batch*i;
        if (tmp_len>data_size)  break;

        Matrix<FLOAT_TYPE> data(raw_data,tmp_len,dim);
        data/=1000;

        Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);

        cct->print_tree_info();
        write_file_log("\n\n",log_file);
        write_file_log(data_file_name,log_file);
        char str_tmp[10];
        sprintf(str_tmp,"%f",leaf_percent);
        write_file_log(",     percent threshold: ",log_file);
        write_file_log(str_tmp,log_file);
        cct->save_tree_info(log_file);
        CYW_TIMER timer;
        timer.start_my_timer();
        Vector<int> indx(20000);
        for (int j=0;j<20000;j++){
            int idx= rand();
            indx[j]=idx%(tmp_len);
        }
        Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);
        cct->kNN(q_points,K);
        timer.stop_my_timer();

        cct->print_kNN_running_time_info();
        char buffer[1000];
        memset(buffer, 0, sizeof(buffer));
        strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
        strcat(buffer,"*     Query points number:");
        my_strcat(buffer,20000);
        strcat(buffer,", and K=");
        my_strcat(buffer,K);
        strcat(buffer,"\n*     GPU Alorithm ");
        timer.strcat_to_buffer(buffer);
        strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
        write_file_log(buffer,log_file);

        delete cct;
    }


}


void test_on_real_data_building_tree(int K, int test_num_each_batch, int batch_num, char *data_file_name, FLOAT_TYPE leaf_percent, FILE* log_file){
    int dim;


    int data_size;

    float *raw_data=read_data(data_file_name," ",&dim, &data_size);

    for (int i=1; i<batch_num+1;i++){
        //---create data set for query
        int tmp_len= test_num_each_batch*i;
        if (tmp_len>data_size)  break;

        Matrix<FLOAT_TYPE> data(raw_data,tmp_len,dim);
        data/=1000;

        Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);

        cct->print_tree_info();
        write_file_log("\n\n",log_file);
        write_file_log(data_file_name,log_file);
        char str_tmp[10];
        sprintf(str_tmp,"%f",leaf_percent);
        write_file_log(",     percent threshold: ",log_file);
        write_file_log(str_tmp,log_file);
        cct->save_tree_info(log_file);

        delete cct;
    }


}

void test_gpu_brute_force_kNN_on_real_data(int K, char *data_file_name,int test_num_each_batch, int batch_num, FLOAT_TYPE leaf_percent, FILE* log_file){
    int dim;


    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);
    write_file_log("\n\n",log_file);
    write_file_log(data_file_name,log_file);

    //data_size=40000;
    //---create data set for query
    Matrix<FLOAT_TYPE> data(raw_data,data_size,dim);
    data;
    //---data.print();


      //create a convex tree
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_LEAF_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_CPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.005);
    Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_LEAF_APPRXIMATE_QP, data,leaf_percent);
    //Convex_Tree<FLOAT_TYPE>* cct= Tree_Helper::create_a_tree<FLOAT_TYPE>(USE_GPU_RECURSIVE_APPRXIMATE_QP, data,(FLOAT_TYPE)0.001);
    cct->print_tree_info();
    cct->save_tree_info(log_file);
    cct->set_batch_size(test_num_each_batch);
    CYW_TIMER timer;
    timer.start_my_timer();
    for (int j=0;j<batch_num; j++){
         /*-------------------------------- Select Data Points From Dataset Randomly -------------------------------*/
        int test_num=test_num_each_batch;
        //char* idx_filename="exp_random_idx_2w(buffer kd tree).txt";
        //float* xTest_idx=read_data(idx_filename,&dim1,&n_size1);
        //FILE* ran_idx_file= fopen("exp_random_idx_2w(convex_tree).txt","a+");
        //write_file_log("\n",ran_idx_file);

        Vector<int> indx(test_num);
        for (int i=0;i<test_num;i++){
            //randomly get an index \in [0,n_size-1]
            int idx= rand();
            indx[i]=idx%data_size;
        }

        Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);
        cct->brute_force_kNN(q_points,K);
        //q_points[1282].print();
        //if ((j%5 )==0)      std::cout<<"j="<<j<<"\n" ;
        /*------------------------------ Select Data Points From Dataset Randomly ---------------------------------*/
    }
    timer.stop_my_timer();
    char buffer[1000];
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
    strcat(buffer,"*     Query points number:");
    my_strcat(buffer,batch_num*test_num_each_batch);
    strcat(buffer,", and K=");
    my_strcat(buffer,K);
    strcat(buffer,"\n*     brute force GPU ");
    timer.strcat_to_buffer(buffer);
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
    write_file_log(buffer,log_file);
    //cct->print_kNN_reult();

    //fclose(log_file);
    //std::cout<<"\n checking....";
    //Vector<NNResult<FLOAT_TYPE>* > results=brute_force_check(data,q_points, K, cct);
    delete cct;
}

Vector<NNResult<FLOAT_TYPE>* > cpu_brute_force(Matrix<FLOAT_TYPE>& data,
                                               Matrix<FLOAT_TYPE>& q_points,
                                               int K)
{
    //std::cout<<"\n cpu brute force is starting...." ;
    int dim=data.ncols();
    long start_t=clock();//GetTickCount();
    Vector<NNResult<FLOAT_TYPE>* > results;
    results.resize(q_points.nrows());

    for (int i=0;i<q_points.nrows();i++){
        FLOAT_TYPE *q = q_points.get_matrix_raw_data()+i*dim;
        Vector<FLOAT_TYPE> tmp_dists_square=pdist2_squre(data, q,dim);
        results[i] = new NNResult<FLOAT_TYPE>();
        results[i]->init(K);
        for (int j=0; j<tmp_dists_square.size();j++){
            results[i]->update(tmp_dists_square[j],j);
        }
        //std::cout<<"\n i="<<i<<" ";
        //results[i]->print();
    }
    long end_t=clock();//GetTickCount();
    //std::cout<<"\n brute force finished, running time is "<< end_t-start_t<<"\ms" ;

}


void test_cpu_brute_force_kNN_on_real_data_2(int K, char *data_file_name,int test_num_each_batch, int batch_num, FILE* log_file){
    int dim;
    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);
    write_file_log("\n\n",log_file);
    write_file_log(data_file_name,log_file);
    Matrix<FLOAT_TYPE> data(raw_data,data_size,dim);
    data/=1000;
    CYW_TIMER timer;
    timer.start_my_timer();
    for (int i=0;i<batch_num;i++){
        Vector<int> indx(test_num_each_batch);
        for (int i=0;i<test_num_each_batch;i++){
            //randomly get an index \in [0,n_size-1]
            int idx= rand();
            indx[i]=idx%data_size;
        }
        //CYW_TIMER timer1;

        Matrix<FLOAT_TYPE> q_points=data.extractRows(indx);
        cpu_brute_force(data, q_points, K);
        std::cout<<"batch="<<i<<"\n";
    }
    timer.stop_my_timer();

    char buffer[1000];
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
    strcat(buffer,"*     Query points number:");
    my_strcat(buffer,batch_num*test_num_each_batch);
    strcat(buffer,", and K=");
    my_strcat(buffer,K);
    strcat(buffer,"\n*     brute force GPU ");
    timer.strcat_to_buffer(buffer);
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
    write_file_log(buffer,log_file);
}




void test_cpu_brute_force_kNN_on_real_data_1(int K_NN, char *data_file_name,int test_num, FILE* log_file){
    int dim;
    int data_size;
    float *raw_data=read_data(data_file_name," ",&dim, &data_size);
    write_file_log("\n\n",log_file);
    write_file_log(data_file_name,log_file);

    CYW_TIMER timer;
    timer.start_my_timer();
    int* idx_k_mins_global_tmp= new int[K_NN*test_num];
    float* dist_k_mins_global_tmp= new float[K_NN*test_num];
    for (int i=0;i<K_NN*test_num;i++){
       dist_k_mins_global_tmp[i]= 3.402823466e+38f;
    }
    float* tmp_data=new float[dim];
    for (int i=0;i<test_num;i++){
        float cur_k_min_dist_square= dist_k_mins_global_tmp[i*K_NN+K_NN-1];
        //random index
        int idx= rand()%data_size;
        int j=0;
        for (j=0;j<dim;j++){
            tmp_data[j]=raw_data[idx*dim+j];
        }
        for (j=0;j<data_size;j++){
            float dist_square_tmp=0;
            for (int k=0;k<dim;k++){
                float tmp=0;
                tmp=raw_data[j*dim+k]-tmp_data[k];
                //compute dist square
                dist_square_tmp+=tmp*tmp;
            }
            //printf("dist_square_tmp =%f, cur_k_min_dist_square=%f \n",dist_square_tmp, cur_k_min_dist_square);
            if (cur_k_min_dist_square> dist_square_tmp){
                //printf("update dist_k_mins_global_tmp...\n");
                int jj = K_NN - 1;
                int current_query_point_index=i;
                dist_k_mins_global_tmp[current_query_point_index*K_NN+jj] = dist_square_tmp;
                int pts_idx =j;
                idx_k_mins_global_tmp[current_query_point_index*K_NN+jj] = pts_idx;

                for(;jj>0;jj--){
                    if(dist_k_mins_global_tmp[current_query_point_index*K_NN+jj-1] > dist_k_mins_global_tmp[current_query_point_index*K_NN+jj]){
                        //printf("new nn found, swap...");
                        float tmp=dist_k_mins_global_tmp[current_query_point_index*K_NN+jj-1];
                        dist_k_mins_global_tmp[current_query_point_index*K_NN+jj-1]=dist_k_mins_global_tmp[current_query_point_index*K_NN+jj];
                        dist_k_mins_global_tmp[current_query_point_index*K_NN+jj]=tmp;

                        //swap indices
                        int tmp_idx=idx_k_mins_global_tmp[current_query_point_index*K_NN+jj-1];
                        idx_k_mins_global_tmp[current_query_point_index*K_NN+jj-1]=idx_k_mins_global_tmp[current_query_point_index*K_NN+jj];
                        idx_k_mins_global_tmp[current_query_point_index*K_NN+jj]=tmp_idx;
                    } else break;
                }
            }

        }
        //std::cout<<"j="<<j;
        if (i%100==0)
            std::cout<<"i="<<i<<"\n";
    }
    timer.stop_my_timer();
    char buffer[1000];
    memset(buffer, 0, sizeof(buffer));
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT ----------------------------------------\n");
    strcat(buffer,"*     Query points number:");
    my_strcat(buffer,test_num);
    strcat(buffer,", and K=");
    my_strcat(buffer,K_NN);
    strcat(buffer,"\n*     brute force GPU ");
    timer.strcat_to_buffer(buffer);
    strcat(buffer,"\n*-----------------------------------------KNN QUERY RESULT----------------------------------------");
    write_file_log(buffer,log_file);
}



int main(int argc, char *const argv[])
{
    int INTEL_GPU_PLATFORM_ID=0;
    int NVIDIA_GPU_PLATFORM_ID=1;
    /*-----------------------------IMPORTANT PARAMETERS-------------------------------------*/
    float leaf_percent_threshold=0.005;
    int batch_size=1000000;
    NOISE_RATE=1;
    int PLATEFORM_ID=INTEL_GPU_PLATFORM_ID;
    //int PLATEFORM_ID=INTEL_GPU_PLATFORM_ID;
    /*-----------------------------IMPORTANT PARAMETERS-------------------------------------*/

    //char *data_file_name="data/house_uni.txt";                   batch_size=1000000;    leaf_percent_threshold=0.001;
    //char *data_file_name="data/blog_uni.txt";                    batch_size=1000000;
    //char *data_file_name="data/pam_uni.txt";                     batch_size=1000000;    leaf_percent_threshold=0.003;
    //char *data_file_name="data/kdd04_norm_uni.txt";
    //char *data_file_name="data/kdd04_16dim.txt";

    //char *data_file_name="data/tom_uni.txt";
    //char *data_file_name="data/hand_posture_uni.txt";

    //char *data_file_name="data/reaction_uni.txt";
    //char *data_file_name="data/reaction_uni_20dim.txt";

    //char* data_file_name="data/USCensus1990_8dim_uni_36w_norm_uni_pca.txt";


    //char* data_file_name="data/mocap.txt";                      batch_size=20000;
    char* data_file_name="data/mopcap_exp_78w_70dim.txt";                      batch_size=20000;
    //char* data_file_name="data/aps.txt";                        batch_size=30000;
    //char* data_file_name="data/BODONI.txt";                     batch_size=40000;
    //char* data_file_name="data/CALIBRI.txt";                    batch_size=40000;

    int K=30;

    FLOAT_TYPE leaf_percent[8];
    leaf_percent[0]=0.01;
    leaf_percent[1]=0.008;
    leaf_percent[2]=0.006;
    leaf_percent[3]=0.004;
    leaf_percent[4]=0.002;
    leaf_percent[5]=0.0009;
    leaf_percent[6]=0.0005;
    leaf_percent[7]=0.0001;

    char log_file_name[1000];
    memset(log_file_name, 0, sizeof(log_file_name));


    if (PLATEFORM_ID==NVIDIA_GPU_PLATFORM_ID)
        strcat(log_file_name,"exp_log_query_time_massive(convex tree)_nvidia_random_query");

    if (PLATEFORM_ID==INTEL_GPU_PLATFORM_ID)
        strcat(log_file_name,"exp_log_query_time_massive(convex tree)_intel_random_query");

    my_strcat(log_file_name, NOISE_RATE);
    strcat (log_file_name,".txt");

    FILE* log_file= fopen(log_file_name,"a+");



    int test_num_each_batch=1000000;

    test_on_real_data_massive(K, test_num_each_batch,1, data_file_name,leaf_percent_threshold,log_file,PLATEFORM_ID, batch_size);
    //test_on_real_data_massive(K, test_num_each_batch,1, data_file_name,0.005,log_file,INTEL_GPU_PLATFORM_ID);
    fclose(log_file);


    //test cardinal
    /*
    FILE* log_file= fopen("exp_log_query_time_with_cardinal(convex tree).txt","a+");
    int test_num_each_batch=200000;
    test_on_real_data_Cardinal( K, test_num_each_batch, 10, data_file_name, 0.001, log_file);
    fclose(log_file);
    */

    /*
    //test tree building
    FILE* log_file= fopen("exp_log_building_tree_time_with_cardinal(convex tree).txt","a+");
    int test_num_each_batch=1906698;
    test_on_real_data_building_tree( K, test_num_each_batch, 1, data_file_name, 0.0003, log_file);
    fclose(log_file);
    */

    /*
    if (PLATEFORM_ID==NVIDIA_GPU_PLATFORM_ID)
        strcat(log_file_name,"exp_log_brute_force_kNN_nvidia_random_query");

    if (PLATEFORM_ID==INTEL_GPU_PLATFORM_ID)
        strcat(log_file_name,"exp_log_brute_force_kNN_intel_random_query");

    my_strcat(log_file_name, NOISE_RATE);
    strcat (log_file_name,".txt");

    FILE* log_file= fopen(log_file_name,"a+");

    test_gpu_brute_force_kNN_on_real_data ( K, data_file_name, 2000,500, leaf_percent_threshold, log_file);
    fclose(log_file);
    */

    /*
    FILE* log_file= fopen("exp_log_brute_force_kNN(cpu).txt","a+");
    test_cpu_brute_force_kNN_on_real_data_1( K, data_file_name, 10000, log_file);
    //test_cpu_brute_force_kNN_on_real_data_2( K, data_file_name, 1000,10, log_file);
    fclose(log_file);
    */
    //test_on_random();
    //int x;
    //std::cin>>x;


    return 0;
}


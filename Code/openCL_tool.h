/*
 *
 * Copyright (C) 2017-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#ifndef OPENCL_TOOL_H_INCLUDED
#define OPENCL_TOOL_H_INCLUDED

/*
/* Error Codes
#define CL_SUCCESS                                   0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST-14
#define CL_COMPILE_PROGRAM_FAILURE                  -15
#define CL_LINKER_NOT_AVAILABLE                     -16
#define CL_LINK_PROGRAM_FAILURE                     -17
#define CL_DEVICE_PARTITION_FAILED                  -18
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64
#define CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define CL_INVALID_COMPILER_OPTIONS                 -66
#define CL_INVALID_LINKER_OPTIONS                   -67
#define CL_INVALID_DEVICE_PARTITION_COUNT           -68
#define CL_INVALID_PIPE_SIZE                        -69
#define CL_INVALID_DEVICE_QUEUE                     -70
*/

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "cyw_types.h"
#define MAX_KERNEL_CONSTANTS_LENGTH 1000
#define MAX_KERNEL_SOURCE_LENGTH 100000
#define USE_DOUBLE 0
using namespace std;

/*
    The data structure of for running openCL.
    0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                          is too large, we can submit subset of query sets to this kernel.
    1 candidate_query_points_indexes    : the indexes of current query points in all query points set
    2 candidate_query_points_set        : the current query points data set
    3 all_sorted_data_set               : all sorted data
    4 sorted_data_set_indexes           : all points indexes in sorted data set
    5 all_nodes_type                    : all_nodes_type[i]=1 means i^th node is a leaf, otherwise isn't.
    6 nodes_num                         : the number of nodes in convex tree
    7 all_ALPHA_set                     : ALPHA set of all nodes
    8 all_BETA_set                      : BETA set of all nodes
    9 all_constrains_num_of_each_nodes  : all_constrains_num_of_each_nodes[i]=j means i^th nodes has j constrains, i.e. has j alphas and betas
   10 all_nodes_offsets_in_all_ALPHA    : the offset of each node in ALPHA
   11 leaf_node_num                     : the number of leaf nodes
   12 leaf_nodes_ori_indexes            : the node index of each leaf node. leaf_node_ori_index[i]=j means the index of i^th leaf node in all nodes is j.
   13 leaf_nodes_start_pos_in_sorted_data_set : specify the start position from which each sorted leave node in sorted data set
   14 pts_num_in_sorted_leaf_nodes      : the length of points saved in each sorted leave node
   15 dist_k_mins_global_tmp            : the K min-distance of all query points,
                                          the length of dist_mins_global_tmp is K* query_points_size
   16 idx_k_mins_global_tmp             : the indexes of the K nearest neighbors,
                                          the length of dist_mins_global_tmp is K* query_points_size
   17 K_NN                              : the value of K
*/
class openCL_stru{
public:
    //ctor
    openCL_stru(){

    }
    cl_int                is_buffer_created=0;
    cl_int                status;
    cl_platform_id        platform;
    cl_platform_id       *platform_ids;
    cl_uint               num_platforms;
    cl_program            program;
    cl_device_id         *devices;
    cl_context            context;
    cl_command_queue      commandQueue ;
    char                 *kernel_filename_brute_computing_distances;
    cl_kernel             do_kNN_kernel;
    cl_kernel             do_brute_force_kNN_kernel;

    /*
        0 candidate_query_points_num        : the number of current candidate query points, in the case of all query points set
                                              is too large, we can submit subset of query sets to this kernel.
        1 candidate_query_points_indexes    : the indexes of current query points in all query points set
        2 candidate_query_points_set        : the current query points data set
        3 candidate_query_points_appr_leaf_node_indexes : the approximate leaf node for candidate query points
        4 all_sorted_data_set               : all sorted data
        5 sorted_data_set_indexes           : all points indexes in sorted data set
        6 tree_struct                       : the tree structure of the whole tree.
        7 all_ALPHA_set                     : ALPHA set of all nodes
        8 all_BETA_set                      : BETA set of all nodes
        9 all_constrains_num_of_each_nodes  : all_constrains_num_of_each_nodes[i]=j means i^th nodes has j constrains, i.e. has j alphas and betas
       10 all_nodes_offsets_in_all_ALPHA    : the offset of each node in ALPHA
       11 leaf_node_num                     : the number of leaf nodes
       12 leaf_nodes_ori_indexes            : the node index of each leaf node. leaf_node_ori_index[i]=j means the index of i^th leaf node in all nodes is j.
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

    /*------------------------------------buffer parameters------------------------------------------------ */
                                                                                        //pram  0: int type
    cl_mem                candidate_query_points_indexes_buffer;                        //parm  1
    cl_mem                candidate_query_points_set_buffer;                            //parm  2
    cl_mem                candidate_approximate_leaf_nodes_buffer ;                     //parm  3
    cl_mem                sorted_data_buffer;                                           //parm  4
    cl_mem                sorted_data_indexes_buffer;                                   //parm  5
    cl_mem                simplified_tree_stru_buffer;                                  //parm  6
    cl_mem                all_ALPHA_set_buffer;                                         //parm  7
    cl_mem                all_BETA_set_buffer;                                          //parm  8
    cl_mem                all_constrains_num_of_each_nodes_buffer;                      //parm  9
    cl_mem                all_nodes_offsets_in_all_ALPHA_buffer;                        //parm 10
                                                                                        //pram 11: int type
    cl_mem                all_leaf_nodes_ancestor_nodes_ids_buffer;                     //parm 12
    cl_mem                sorted_leaf_nodes_start_pos_in_sorted_data_buffer;            //parm 13
    cl_mem                pts_num_in_sorted_leaf_nodes_buffer;                          //parm 14
    cl_mem                dist_k_mins_global_buffer;                                    //parm 15
    cl_mem                idx_k_mins_global_buffer;                                     //parm 16
                                                                                        //parm 17: K_NN int type
    //this is used to save the distance computation times
    cl_mem                dist_computation_times_arr_in_device_buffer;                  //parm 18
    cl_mem                quadprog_times_arr_in_device_buffer;                          //parm 19:
    cl_mem                dist_computation_times_arr_in_quadprog_buffer;                //parm 20

    /*------------------------------------buffer parameters------------------------------------------------ */

    void free_resource(){
        if (platform_ids != NULL)
        {
            free(platform_ids);
            platform_ids = NULL;
        }

        if (devices != NULL)
        {
            free(devices);
            devices = NULL;
        }
        free(kernel_filename_brute_computing_distances);
        cl_int status;
                                                                                           //pram  0: int type
        status = clReleaseMemObject(candidate_query_points_indexes_buffer);                //parm  1
        status = clReleaseMemObject(candidate_query_points_set_buffer);                    //parm  2
        status = clReleaseMemObject(candidate_approximate_leaf_nodes_buffer);              //pram  3
        status = clReleaseMemObject(sorted_data_buffer);                                   //parm  4
        status = clReleaseMemObject(sorted_data_indexes_buffer);                           //parm  5
        status = clReleaseMemObject(simplified_tree_stru_buffer);                          //parm  6
        status = clReleaseMemObject(all_ALPHA_set_buffer);                                 //parm  7
        status = clReleaseMemObject(all_BETA_set_buffer);                                  //parm  8
        status = clReleaseMemObject(all_constrains_num_of_each_nodes_buffer);              //parm  9
        status = clReleaseMemObject(all_nodes_offsets_in_all_ALPHA_buffer);                //parm 10
                                                                                           //pram 11: int type
        status = clReleaseMemObject(all_leaf_nodes_ancestor_nodes_ids_buffer);             //parm 12
        status = clReleaseMemObject(sorted_leaf_nodes_start_pos_in_sorted_data_buffer);    //parm 13
        status = clReleaseMemObject(pts_num_in_sorted_leaf_nodes_buffer);                  //parm 14
        if (is_buffer_created){
            status = clReleaseMemObject(dist_k_mins_global_buffer);                            //parm 15
            status = clReleaseMemObject(idx_k_mins_global_buffer);                             //parm 16

                                                                                               //parm 17: K_NN int type, no need to release
            status = clReleaseMemObject(dist_computation_times_arr_in_device_buffer);          //parm 18
            status = clReleaseMemObject(quadprog_times_arr_in_device_buffer);                  //parm 19
            status = clReleaseMemObject(dist_computation_times_arr_in_quadprog_buffer);        //parm 20
        }

    }

    //dctor
    ~openCL_stru(){
        free_resource();
    }
};


/**
 * Helper function that checks for an OpenCL error
 *
 * @param err The OpenCL error code (int)
 * @param *file The source code file
 * @param line The associated line
 *
 */
void check_cl_error(cl_int err,
		const char *file,
		int line) {

	if (err != CL_SUCCESS) {
		printf("An OpenCL error with code %i in file %s and line %i occurred ...\n", err, file, line);
		//exit(1);
	}

}

//read kernel source file
void readfile_cyw(char *str,char *Fname,unsigned long *size)
{
	FILE*filepointer;
	int wordnum, is_end;
	char read_char;
	//printf("Enter the file name:");
	filepointer=fopen(Fname,"r");
	wordnum=0;
	is_end=fscanf(filepointer,"%c",&read_char);
	while (is_end!=EOF)
	{
		if(isupper(read_char)|| islower(read_char))
		{
			str[wordnum]=read_char;wordnum++;
		}
		else if(read_char==' '||read_char=='\n')
		{
			str[wordnum]=read_char;wordnum++;
		}
		else
		{
			str[wordnum]=read_char;wordnum++;
		}
		is_end=fscanf(filepointer,"%c",&read_char);

	}
	str[wordnum]=0x00;
	*size=wordnum;
	fclose(filepointer);
}

 /**
 * Generates an OpenCL kernel from a source string.
 *
 * @param context The OpenCL context
 * @param device The OpenCL device
 * @param *kernel_constants Pointer to string that contains kernel constants that shall be added to the compiled kernel
 * @param *kernel_filename Pointer to string containing the kernel code
 * @param *kernel_name Pointer to string containing the kernel name
 */
cl_kernel make_kernel_from_file(openCL_stru& os, char *kernel_constants, const char *kernel_name) {
	cl_int err;
	cl_kernel kernel;
    cl_context context=os.context;
    char *kernel_filename=os.kernel_filename_brute_computing_distances;
    cl_device_id device= os.devices[0];
	//read kernel sources
	//char *kernel_source;
	char kernel_source[50000] ;
	unsigned long size;
	//printf(kernel_filename);
	readfile_cyw(kernel_source,kernel_filename,&size);
	//readfile(kernel_filename, &kernel_source, &size);

	if (size > MAX_KERNEL_SOURCE_LENGTH - MAX_KERNEL_CONSTANTS_LENGTH) {
		printf("Kernel source file too long ...\n");
		exit(1);
	}

	char outbuff[MAX_KERNEL_SOURCE_LENGTH] = "";
	strcat(outbuff, kernel_constants);
	strncat(outbuff, kernel_source, size);
	strcat(outbuff, "\0");
    //printf(outbuff);
	const char *outbuff_new = (const char*) outbuff;
	size_t outbuff_new_length = (size_t) strlen(outbuff_new);

	// generate program
	os.program = clCreateProgramWithSource(context, 1, &outbuff_new,
			                               &outbuff_new_length, &err);

	check_cl_error(err, __FILE__, __LINE__);

	// generate for all devices
	err = clBuildProgram(os.program, 0, NULL, NULL, NULL, NULL);

	// print build log if needed
	if (err != CL_SUCCESS) {
		printf("Error while compiling file %s\n", kernel_filename);
		//print_build_information(program, device);
		size_t len;
        char buffer[8 * 1024];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(os.program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
	}
	check_cl_error(err, __FILE__, __LINE__);

	// generate kernel
	kernel = clCreateKernel(os.program, kernel_name, &err);
		// print build log if needed
	if (err != CL_SUCCESS) {
		printf("Error while compiling file %s\n", kernel_filename);
		//print_build_information(program, device);
		size_t len;
        char buffer[8 * 1024];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(os.program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
	}else{
        printf("succeed to build program executable!\n");
	}
	check_cl_error(err, __FILE__, __LINE__);

	// release program
	//clReleaseProgram(program);
	// release memory
	free(kernel_source);

	return kernel;
}


// convert the kernel file into a string
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout<<"Error: failed to open file\n:"<<filename<<endl;
    return -1;
}

//Getting platforms and choose an available one.
int getPlatform(cl_platform_id &platform)
{
    platform = NULL;             //the chosen platform
    cl_uint numPlatforms;        //the NO. of platforms

    //call clGetPlatformIDs twice, the first time retrieves the number of platform
    //the second time retrieves an available platform id.
    cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        cout<<"Error: Getting platforms!"<<endl;
        return -1;
    }

    //For clarity, choose the first available platform.
    if(numPlatforms > 0)
    {
        cl_platform_id* platforms =(cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
        //the second time retrieves an available platform id.
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
    }
    else
        return -1;
}

//Getting platforms and choose an available one.
int getPlatform_ids(cl_platform_id* &platform_ids, cl_uint &numPlatforms)
{
    //call clGetPlatformIDs twice, the first time retrieves the number of plateforms£¬
    //the second time retrieves an available platform id.
    cl_int   status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        cout<<"Error: Getting platforms!"<<endl;
        return -1;
    }
    /**For clarity, choose the first available platform. */
    if(numPlatforms > 0)
    {
        //the second time retrieves an available platform id.
        platform_ids =  (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platform_ids, NULL);
        if (status != CL_SUCCESS)
        {
            cout<<"Error: Getting platforms!"<<endl;
            return -1;
        }
    }
    else
        return -1;

    return 1;
}

//Step 3:Query the platform and choose the first GPU device if has one.
cl_device_id *getCl_device_id(cl_platform_id &platform)
{
    cl_uint numDevices = 0;
    cl_device_id *devices=NULL;
    //call clGetPlatformIDs twice, first time retrieves the number of devices
    //the second time retrieves an available devices id.
    cl_int    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices > 0) //GPU available.
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        //the second time retrieves an available platform id.
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }
    return devices;
}

//Step 2:Query the platform and return all devices.
cl_device_id *getCl_device_ids(cl_platform_id &platform, cl_uint &num_devices)
{
    num_devices = 0;
    cl_device_id *devices=NULL;
    //call clGetPlatformIDs twice, first time retrieves the number of devices
    //the second time retrieves an available device id.
    cl_int    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (num_devices > 0) //GPU available.
    {
        //the second time retrieves an available device id.
        devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    }
    return devices;
}


#endif // OPENCL_TOOL_H_INCLUDED

#ifndef DATA_PROCESSOR_H_INCLUDED
#define DATA_PROCESSOR_H_INCLUDED


int get_dim(char* s, char* delims){
    char *val_str = NULL;
    val_str = strtok( s, delims );
    int dim=0;
    while( val_str != NULL ) {
        dim++;
        val_str = strtok( NULL, delims );
    }
    return dim;
}

float*  get_data(char* s, int dim,char* delims)
{
    float* temp= (float*) malloc (dim*sizeof(float));
    char *val_str = NULL;
    val_str = strtok( s, delims );
    int counter=0;
    while( val_str != NULL ) {
        //printf(val_str);
        temp[counter]=atof(val_str);
        counter++;
        val_str = strtok( NULL, delims );
    }
    return temp;
}

void read_data_dim_size(char* filename, int* data_dim, int* data_size, char* delims){
    int n_size=0;
    int dim=0;

    char s[10000];
    freopen(filename,"r",stdin);
    while(gets(s))
    {
        if (dim==0)
           dim=get_dim(s,delims);
        n_size ++;
    }
    *data_dim=dim;
    *data_size=n_size;
    fclose(stdin);
}

float* read_data(char* filename, char* delims){
    int m = 0;
    int dim, n_size;
    read_data_dim_size(filename,&dim, &n_size, delims);

    float* data= (float*) malloc (n_size*dim*sizeof(float));
    freopen(filename,"r",stdin);
    int counter=0;
    char s[10000];
    while(gets(s))
    {
        float* tmp_data= get_data( s, dim,delims);
        memcpy(data+counter*dim,tmp_data,dim*sizeof(float));
        /*
            for (int i=0;i<dim;i++){
               *(data+counter*dim+i)= tmp_data[i];
               //printf("\n%f, ",*(data+counter*dim+i));
            }
        */
        counter++;
        free(tmp_data);
    }
    fclose(stdin);

    return data;
}

float* read_data(char* filename, char* delims, int* dim, int* data_size){
    int m = 0;
    read_data_dim_size(filename,dim, data_size, delims);

    float* data= (float*) malloc ((*data_size)*(*dim)*sizeof(float));
    freopen(filename,"r",stdin);
    int counter=0;
    char s[10000];
    while(gets(s))
    {
        float* tmp_data= get_data( s,*dim,delims);
        memcpy(data+counter*(*dim),tmp_data,(*dim)*sizeof(float));
        /*
            for (int i=0;i<dim;i++){
               *(data+counter*dim+i)= tmp_data[i];
               //printf("\n%f, ",*(data+counter*dim+i));
            }
        */
        counter++;
        free(tmp_data);
    }
    fclose(stdin);

    return data;
}

#endif // DATA_PROCESSOR_H_INCLUDED

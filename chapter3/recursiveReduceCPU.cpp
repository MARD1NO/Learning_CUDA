#include "iostream"
using namespace std; 

int recursiveReduce(float *data, int const size){
    // teminate recur
    if(size == 1){
        return data[0];
    }

    int stride = size / 2;

    for(int i = 0; i < stride; i++){
        data[i] += data[i+stride];
    }
    return recursiveReduce(data, stride);
}

int main(){
    float arr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int size = 8;
    float sum = recursiveReduce(arr, size);
    cout<<"Sum is : "<<sum<<endl;
    return 0;
}
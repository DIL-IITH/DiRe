#!/bin/bash
for i in {0..90..10}
do 
    echo $i
    sh recover_cifar_regularized.sh 0 10 cuda:0 0.1 $i 
    sh recover_cifar_regularized.sh 50 10 cuda:0 0.1 $i 
done 

#!/bin/sh
name=ffnet
director=log_ffnet
gpu_num=2
##for dataset in "1,5" "1,6" "3,6" "3,9" "3,8"
#for dataset in "3,8"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
##    echo "${name}_${dataset}_$i"
#    CUDA_VISIBLE_DEVICES=$gpu_num python -u exe_mnist.py -nn "4, 2" -bin -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -chk -chkname  ${name}_${dataset}_$i> ${director}/${name}_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done


for dataset in "0,3,6" "1,3,6" "2,4,8" "3,6,9"
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
#    echo "${name}_${dataset}_$i"
    CUDA_VISIBLE_DEVICES=$gpu_num python -u exe_mnist.py -qa "-1 -1 1 1 1 -1 1 -1, -1 -1 -1" -nn "8, 3" -bin -wn -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -chk -chkname  ${name}_${dataset}_$i> ${director}/${name}_${dataset}_$i 2>&1 &
    if [ $((i%5)) -eq 0 ]
    then
      wait
    fi
  done
done



for dataset in "0,3,6,9" "1,3,5,7" "1,5,7,9" "2,4,6,8"
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
#    echo "${name}_${dataset}_$i"
    CUDA_VISIBLE_DEVICES=$gpu_num python -u exe_mnist.py -qa "1 -1 1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 1 1, -1 -1 -1 -1" -nn "16, 4" -bin -wn -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -chk -chkname  ${name}_${dataset}_$i> ${director}/${name}_${dataset}_$i 2>&1 &
    if [ $((i%5)) -eq 0 ]
    then
      wait
    fi
  done
done


for dataset in "0,1,2,3,4" "1,2,3,4,5" "0,1,3,6,9" "3,4,5,6,7"
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
#    echo "${name}_${dataset}_$i"
    CUDA_VISIBLE_DEVICES=$gpu_num python -u exe_mnist.py -qa "1 -1 1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 1 1, -1 -1 -1 -1 -1" -nn "16, 5" -bin -wn -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -chk -chkname  ${name}_${dataset}_$i> ${director}/${name}_${dataset}_$i 2>&1 &
    if [ $((i%2)) -eq 0 ]
    then
      wait
    fi
  done
done


# "1,5" "1,6" "3,6" "3,9" "3,8"
#"0,3,6" "1,3,6" "2,4,8" "3,6,9"
#"0,3,6,9" "1,3,5,7" "1,5,7,9" "2,4,6,8"
#"0,1,2,3,4" "1,2,3,4,5" "0,1,3,6,9" "3,4,5,6,7"

#!/bin/sh

#for dataset in "0,1,2,3,4" "1,2,3,4,5" "0,1,3,6,9" "3,4,5,6,7"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
##    echo "qfnet_p_${dataset}_$i"
#    CUDA_VISIBLE_DEVICES=3 python -u exe_mnist.py -nn "16, 5" -wn -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_${dataset}_$i> log/qfnet_p_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done


#for dataset in "1,5" "1,6" "3,6" "3,9" "3,8"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
#    CUDA_VISIBLE_DEVICES=2 python -u exe_mnist.py -nn "4, 2" -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_wobn_${dataset}_$i> log/qfnet_p_wobn_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done

#
#for dataset in "0,1,2,3,4" "1,2,3,4,5" "0,1,3,6,9" "3,4,5,6,7"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
#    CUDA_VISIBLE_DEVICES=2 python -u exe_mnist.py -nn "16, 5" -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_wobn_${dataset}_$i> log/qfnet_p_wobn_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done



#
#for dataset in "0,3,6,9" "1,3,5,7" "1,5,7,9" "2,4,6,8"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
#    CUDA_VISIBLE_DEVICES=3 python -u exe_mnist.py -nn "16, 4" -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_wobn_${dataset}_$i> log/qfnet_p_wobn_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done


#
#for dataset in "0,3,6" "1,3,6" "2,4,8" "3,6,9"
#do
#  for i in 1 2 3 4 5 6 7 8 9 10
#  do
#    CUDA_VISIBLE_DEVICES=1 python -u exe_mnist.py -nn "8, 3" -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_wobn_${dataset}_$i> log/qfnet_p_wobn_${dataset}_$i 2>&1 &
#    if [ $((i%5)) -eq 0 ]
#    then
#      wait
#    fi
#  done
#done


for dataset in "0,1,3,6,9"
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python -u exe_mnist.py --device cpu -nn "8, 5" -wn -qt -c $dataset -s 16 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" -ppd -dp "../../pytorch/data/MNIST/processed" -chk -chkname  qfnet_p_${dataset}_star_$i> log/qfnet_p_${dataset}_star_$i 2>&1 &
    if [ $((i%5)) -eq 0 ]
    then
      wait
    fi
  done
done
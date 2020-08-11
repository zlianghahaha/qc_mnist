CUDA_VISIBLE_DEVICES=3 python -u exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -ql 0.0001 -e 30 -m "10, 20" -chk > log/qnet+_1_36.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=3 python -u exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -ql 0.0001 -e 30 -m "10, 20" -chk > log/qnet+_2_36.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=3 python -u exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -ql 0.0001 -e 30 -m "10, 20" -chk > log/qnet+_3_36.res 2>&1 &

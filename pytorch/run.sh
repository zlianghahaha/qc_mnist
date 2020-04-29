CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt1.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt2.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt3.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q1.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q2.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q3.res 2>&1 &
sleep 1
wait

CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq1.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq2.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq3.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq1.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq2.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq3.res 2>&1 &
sleep 1
wait


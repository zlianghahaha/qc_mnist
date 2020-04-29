CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
wait

CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk &
sleep 1
wait


CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt1_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt2_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -qt -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt3_38.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q1_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q2_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q3_38.res 2>&1 &
sleep 1
wait

CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq1_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq2_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq3_38.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq1_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq2_38.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -nq -c "3, 8" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq3_38.res 2>&1 &
sleep 1
wait



CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -qt -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt1_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -qt -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt2_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -qt -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnqt3_136.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q1_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q2_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/q3_136.res 2>&1 &
sleep 1
wait

CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq1_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq2_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -tb 16 -nn "8, 3" -wn -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/wnnq3_136.res 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq1_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq2_136.res 2>&1 &
sleep 1
CUDA_VISIBLE_DEVICES=1 python exe_mnist.py -tb 16 -nn "8, 3" -nq -c "1, 3, 6" -s 4 -l 0.1 -e 30 -m "10, 20" -chk > log/nq3_136.res 2>&1 &
sleep 1
wait


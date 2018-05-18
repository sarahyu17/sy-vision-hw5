run_all: run_23 test_batch_size test_learning_rate test_adjust_learning_rate test_pre_data_aug

run_23: FORCE
	python main.py --model CoolNet --epochs 50 --cuda True

test_batch_size: FORCE
	python main.py --model CoolNet --epoch 50 --batchSize 64 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 128 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 256 --cuda True

test_learning_rate: FORCE
	python main.py --lr 10 --model CoolNet --epoch 50 --cuda True

	python main.py --lr 0.1 --model CoolNet --epoch 50 --cuda True

	python main.py --lr 0.01 --model CoolNet --epoch 50 --cuda True

	python main.py --lr 0.0001 --model CoolNet --epoch 50 --cuda True


test_adjust_learning_rate: FORCE
	python main.py --model CoolNet --epoch 150 --batchSize 256 --cuda True

test_pre_data_aug: FORCE
	python3 main.py --epochs 200 --model CoolNet --lr 0.01 --cuda True


# Phony target to force clean
FORCE: ;
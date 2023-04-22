from setting import environment, constant
from util import path, generator
from nn import nn
import argparse

### python main.py --dip=example --tolabel ?
# Step 1:
### python main.py --dataset=example --dip=example --augmentation=0000
# Step 2: train
### python main.py --dataset=example --gpu --train
# Step3: test
### python main.py --dataset=example --arch=example --dip=example --gpu --test

### python main.py --dataset=rice --dip=example --arch=example --augmentation=100  --gpu --train
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tolabel", help="Preprocess images to create labels (out/tolabel)", action="store_true", default=False)
	parser.add_argument("--augmentation", help="Dataset augmentation (pass quantity)", type=int)
	parser.add_argument("--dataset", help="Dataset name", type=str, default=constant.DATASET)
	parser.add_argument("--train", help="Train", action="store_true", default=False)
	parser.add_argument("--test", help="Predict", action="store_true", default=False)
	parser.add_argument("--arch", help="Neural Network architecture", type=str, default=constant.MODEL)
	parser.add_argument("--dip", help="Method for image processing", type=str, default=constant.IMG_PROCESSING)
	parser.add_argument("--gpu", help="Enable GPU mode", action="store_true", default=False)
	args = parser.parse_args()

	environment.setup(args)
	exist = lambda x: len(x)>0 and path.exist(path.data(x, mkdir=False))

	if (args.tolabel):
		generator.tolabel()

	elif args.dataset is not None and exist(args.dataset):

		if (args.augmentation):
			generator.augmentation(args.augmentation)

		elif (args.train):
			nn.train()

		elif (args.test):
			nn.test()
	else:
		print("\n>> Dataset not found\n")

if __name__ == "__main__":
	main()

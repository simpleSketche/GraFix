from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from datetime import datetime
import os

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import logging
import dataset
from model import *
from utils import binary_classification_accracy
import matplotlib.pyplot as plt



def main(args):

	print(f"Using device: {args.device}")
	args.ckpt_dir = os.path.join(args.ckpt_dir, datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
	os.mkdir(args.ckpt_dir)
	logger = get_loggings(args.ckpt_dir)

	# dataset, split
	d = dataset.BoxDataset(root=args.data_path, data_num=args.data_num)
	logger.info(f"Dataset length: {len(d)}")
	data_num = len(d)
	train_ratio, valid_ratio, test_ratio = 0.8, 0.2, 0.0
	train_num, valid_num = int(data_num*train_ratio), int(data_num*valid_ratio)
	test_num = data_num - train_num - valid_num
	train_dataset, valid_dataset, test_dataset = random_split(d, [train_num, valid_num, test_num])
	# train_dataset, valid_dataset, test_dataset = d.split()
	logger.info(f"train, valid, test dataset len: {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}")

	# crecate DataLoader for train / dev datasets
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	encoder_args = {"input_dim": 5, "output_dim": args.hidden_dim}
	encoder = globals()[args.encoder_name](**encoder_args).to(args.device)
	model = globals()[args.model_name](encoder).to(args.device)
	logger.info(model)

	# init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	best_eval_loss = np.inf

	for epoch in range(1, args.num_epoch+1):
		# Training loop - iterate over train dataloader and update model weights
		model.train()
		loss_train, pos_acc_train, neg_acc_train, iter_train = 0, 0, 0, 0
		for batch in train_loader:
			batch = batch.to(args.device)
			z = model.encode(batch.x, batch.edge_index)
			loss = model.recon_loss(z, batch.edge_index, batch.batch, batch.neg_edge_index)
			loss += model.kl_loss()

			# calculate loss and update parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# accumulate loss, accuracy
			iter_train += 1
			loss_train += loss.item()
			pos_acc, neg_acc = model.test(z, batch.edge_index, batch.batch, batch.neg_edge_index)
			pos_acc_train += pos_acc
			neg_acc_train += neg_acc
		
		loss_train /= iter_train
		pos_acc_train /= iter_train
		neg_acc_train /= iter_train

		# Evaluation loop - calculate accuracy and save model weights
		model.eval()
		with torch.no_grad():
			loss_eval, pos_acc_eval, neg_acc_eval, iter_eval = 0, 0, 0, 0
			for batch in valid_loader:
				batch = batch.to(args.device)
				z = model.encode(batch.x, batch.edge_index)
				loss = model.recon_loss(z, batch.edge_index, batch.batch)
				loss += model.kl_loss()

				# accumulate loss, accuracy
				iter_eval += 1
				loss_eval += loss.item()
				pos_acc, neg_acc = model.test(z, batch.edge_index, batch.batch, batch.neg_edge_index)
				pos_acc_eval += pos_acc
				neg_acc_eval += neg_acc
				
			loss_eval /= iter_eval
			pos_acc_eval /= iter_eval
			neg_acc_eval /= iter_eval

		logger.info(f"epoch: {epoch:4d}, train_acc_pos: {pos_acc_train:.4f}, train_acc_neg: {neg_acc_train:.4f} eval_acc_pos: {pos_acc_eval:.4f}, eval_acc_neg: {neg_acc_eval:.4f}, train_loss: {loss_train:.4f}, eval_loss: {loss_eval:.4f}")

		# save model
		if loss_eval < best_eval_loss:
			best_eval_loss = loss_eval
			logger.info(f"Trained model saved, eval loss: {best_eval_loss:.4f}")
			torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "model.pt"))
		
		


def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument(
		"--data_path",
		type=Path,
		help="Directory to the dataset.",
		default="./data/",
	)

	parser.add_argument(
		"--ckpt_dir",
		type=Path,
		help="Directory to save the model file.",
		default="./Results/",
	)

	# dataset
	parser.add_argument("--data_num", type=int, default=5000)

	# model
	parser.add_argument("--encoder_name", type=str, default="VariationalGCNEncoder")
	parser.add_argument("--model_name", type=str, default="VGAE")
	parser.add_argument("--hidden_dim", type=int, default=64)

	# optimizer
	parser.add_argument("--lr", type=float, default=1e-3)

	# data loader
	parser.add_argument("--batch_size", type=int, default=256)

	# training
	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
	)
	parser.add_argument("--num_epoch", type=int, default=100)

	args = parser.parse_args()
	return args


def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='graphVAE')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


if __name__ == "__main__":
	args = parse_args()
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)
	main(args)
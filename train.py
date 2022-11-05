from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from datetime import datetime
import os
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

import dataset
from model import *
from utils import binary_classification_accracy
from utils import R2_score





def main(args):

	print(f"Using device: {args.device}")
	args.ckpt_dir = os.path.join(args.ckpt_dir, datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
	os.mkdir(args.ckpt_dir)
	logger = get_loggings(args.ckpt_dir)
	logger.info(args)

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

	encoder_args = {"input_dim": 10, "hidden_dim": args.hidden_dim}
	decoder_args = {"hidden_dim": args.hidden_dim, "output_dim": 3}
	encoder = globals()[args.encoder_name](**encoder_args).to(args.device)
	decoder = globals()[args.decoder_name](**decoder_args).to(args.device)
	model = globals()[args.model_name](encoder, decoder).to(args.device)
	logger.info(model)

	# VAE loss function (reconstruction loss + KL_Divergence loss)
	def loss_fn(recon_x, x, mu, log_var, i):
		BCE = torch.nn.functional.binary_cross_entropy(recon_x[:, 0], x[:, 2])
		MSE = torch.nn.functional.mse_loss(recon_x[:, 1:3], x[:, 3:5])
		KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * args.kl_loss_ratio
		if i == 0:
			logger.info(f"BCE: {BCE:5.3f}, MSE: {MSE:5.3f}, KLD: {KLD:5.3f}")
		return BCE + MSE + KLD

	# init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	best_eval_loss = np.inf

	for epoch in range(1, args.num_epoch+1):
		# Training loop - iterate over train dataloader and update model weights
		model.train()
		loss_train, acc_reg_train, acc_cls_train, iter_train = 0, 0, 0, 0
		for i, batch in enumerate(train_loader):
			batch = batch.to(args.device)
			recon_x, mu, log_var, z = model(batch.x, batch.edge_index)

			# calculate loss and update parameters
			loss = loss_fn(recon_x, batch.x, mu, log_var, i)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# accumulate loss, accuracy
			iter_train += 1
			loss_train += loss.item()
			acc_reg_train += R2_score(recon_x[:, 1:3], batch.x[:, 3:5])
			acc_cls_train += binary_classification_accracy(recon_x[:, 0], batch.x[:, 2])
		
		loss_train /= iter_train
		acc_reg_train /= iter_train
		acc_cls_train /= iter_train

		# Evaluation loop - calculate accuracy and save model weights
		model.eval()
		with torch.no_grad():
			loss_eval, acc_reg_eval, acc_cls_eval, iter_eval = 0, 0, 0, 0
			for batch in valid_loader:
				batch = batch.to(args.device)
				recon_x, mu, log_var, z = model(batch.x, batch.edge_index)
				loss = loss_fn(recon_x, batch.x, mu, log_var, None)

				# accumulate loss, accuracy
				iter_eval += 1
				loss_eval += loss.item()
				acc_reg_eval += R2_score(recon_x[:, 1:3], batch.x[:, 3:5])
				acc_cls_eval += binary_classification_accracy(recon_x[:, 0], batch.x[:, 2])
				
			loss_eval /= iter_eval
			acc_reg_eval /= iter_eval
			acc_cls_eval /= iter_eval

		logger.info(f"epoch: {epoch:4d}, train_reg_acc: {acc_reg_train:.4f}, eval_reg_acc: {acc_reg_eval:.4f}, train_cls_acc: {acc_cls_train:.4f}, eval_cls_acc: {acc_cls_eval:.4f}, train_loss: {loss_train:.4f}, eval_loss: {loss_eval:.4f}")

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
	parser.add_argument("--encoder_name", type=str, default="VariationalGraphEncoder")
	parser.add_argument("--decoder_name", type=str, default="VariationalGraphDecoder")
	parser.add_argument("--model_name", type=str, default="VGAE")
	parser.add_argument("--hidden_dim", type=int, default=256)

	# optimizer
	parser.add_argument("--kl_loss_ratio", type=float, default=2e-3)
	parser.add_argument("--lr", type=float, default=1e-3)

	# data loader
	parser.add_argument("--batch_size", type=int, default=32)

	# training
	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
	)
	parser.add_argument("--num_epoch", type=int, default=500)

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
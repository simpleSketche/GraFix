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

from dataset import *
from model import *
from utils import binary_classification_accracy, R2_score





def main(args):

	print(f"Using device: {args.device}")
	args.ckpt_dir = args.ckpt_dir / datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)
	logger = get_loggings(args.ckpt_dir)
	logger.info(args)

	# dataset, split
	d = RoomAsNodeDataset(root=args.data_path, data_num=args.data_num)
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

	model_args = {"input_dim": 8+GRAPH_BASED_FEATURE_DIM, "hidden_dim": args.hidden_dim, "output_dim": 8,
			 	  "num_layers": args.num_layers, "message_passing": args.message_passing}
	model = globals()[args.model_name](**model_args).to(args.device)
	logger.info(model)

	# init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=10, min_lr=1e-4)
	criterion = torch.nn.MSELoss()
	best_eval_loss = np.inf

	for epoch in range(1, args.num_epoch+1):
		# Training loop - iterate over train dataloader and update model weights
		model.train()
		loss_train, acc_train, iter_train = 0, 0, 0
		for i, batch in enumerate(train_loader):
			batch = batch.to(args.device)
			pred = model(batch.x, batch.edge_index)

			# calculate loss and update parameters
			loss = criterion(pred, batch.y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# accumulate loss, accuracy
			iter_train += 1
			loss_train += loss.item()
			acc_train += R2_score(pred, batch.y)
		
		loss_train /= iter_train
		acc_train /= iter_train

		# Evaluation loop - calculate accuracy and save model weights
		model.eval()
		with torch.no_grad():
			loss_eval, acc_eval, iter_eval = 0, 0, 0
			for batch in valid_loader:
				batch = batch.to(args.device)
				pred = model(batch.x, batch.edge_index)
				loss = criterion(pred, batch.y)

				# accumulate loss, accuracy
				iter_eval += 1
				loss_eval += loss.item()
				acc_eval += R2_score(pred, batch.y)
				
			loss_eval /= iter_eval
			acc_eval /= iter_eval

		# learning rate scheduler 
		scheduler.step(loss_train)

		logger.info(f"epoch: {epoch:4d}, train_acc: {acc_train:.4f}, eval_acc: {acc_eval:.4f}, train_loss: {loss_train:.4f}, eval_loss: {loss_eval:.4f}")

		# save model
		if loss_eval < best_eval_loss:
			best_eval_loss = loss_eval
			logger.info(f"Trained model saved, eval loss: {best_eval_loss:.4f}")
			torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "model.pt"))
		
	# save prediction
	save_pred_train_path = args.ckpt_dir / "Prediction_train"
	save_pred_train_path.mkdir(parents=True, exist_ok=True)
	for i in range(args.save_num):
		save_path = save_pred_train_path / f"pred_{i}.pt"
		logger.info(f"saving prediction: {save_path}")
		graph = train_dataset[i].to(args.device)
		pred = model(graph.x, graph.edge_index)
		output_graph(save_path, graph, pred)

	save_pred_valid_path = args.ckpt_dir / "Prediction_valid"
	save_pred_valid_path.mkdir(parents=True, exist_ok=True)
	for i in range(args.save_num):
		save_path = save_pred_valid_path / f"pred_{i}.pt"
		logger.info(f"saving prediction: {save_path}")
		graph = valid_dataset[i].to(args.device)
		pred = model(graph.x, graph.edge_index)
		output_graph(save_path, graph, pred)






def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument(
		"--data_path",
		type=Path,
		help="Directory to the dataset.",
		default="./data/room_base_graph/",
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
	parser.add_argument("--model_name", type=str, default="GNN")
	parser.add_argument("--message_passing", type=str, default="GAT")
	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--num_layers", type=int, default=3)

	# optimizer
	parser.add_argument("--lr", type=float, default=1e-3)

	# data loader
	parser.add_argument("--batch_size", type=int, default=64)

	# training
	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
	)
	parser.add_argument("--num_epoch", type=int, default=1500)
	parser.add_argument("--save_num", type=int, default=50)

	args = parser.parse_args()
	return args


def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='RoomAsNodeGNN')
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
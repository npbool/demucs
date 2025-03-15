python3 train.py \
	--ds /root/autodl-tmp/data \
	--mixed_precision bf16 \
	--lr 1e-4 \
	--loss l1 \
	--batch_size 6 \
	--gradient_accumulation_steps 4 \
	--gradient_checkpoint \
	--output_dir train_v3 \
	 --resume_from_checkpoint latest \
	--checkpointing_steps 500 \
	--validate_steps 500
	#--init_from model.safetensors \

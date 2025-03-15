rsync -azv \
	--exclude ".git" \
	--exclude "train_v*" \
	--exclude "checkpoints" \
	. autodl:/root/code


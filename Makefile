# Makefile

pretrain:
	uv run python main.py --config_path=config/contrastive-pretrain-config.json --no_ray

eval:
	uv run python main.py --config_path=config/eval-config.json --no_ray

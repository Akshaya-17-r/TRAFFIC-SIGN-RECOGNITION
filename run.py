import argparse
from src.train import train
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','eval'], default='train')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    if args.mode == 'train':
        train(config_path=args.config)
    else:
        evaluate(config_path=args.config, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()

import argparse, torch
from data.cifar100_split import get_task_loaders
from models.resnet18 import ResNet18CIFAR
from trainers.trainer import train_one_task

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=0, help="0..4")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--method", type=str, default="ER", choices=["ER","EWC"])
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, classes = get_task_loaders(args.task, batch_size=args.batch)
    print(f"Task {args.task}, classes={classes[0]}..{classes[-1]}")
    model = ResNet18CIFAR(num_classes=100)
    train_one_task(model, train_loader, test_loader, device=device, epochs=args.epochs, lr=0.1)

if __name__ == "__main__":
    main()


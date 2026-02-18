"""
ResNet9 CIFAR-10 Training Baseline

A simple, un-optimized training script intended as a profiling baseline.
No torch.compile, no AMP, no prefetching tricks.

Run:  python resnet9_cifar10.py
"""

import time

import torch
import torch.nn as nn
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

from model import ResNet9
from data import get_dataloaders
from train import train_one_epoch, evaluate

PROFILE_DIR = "./log/profiler"


def main():
    epochs = 10
    batch_size = 128
    lr = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = ResNet9().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
    )

    # Profile epoch 1: skip 1 batch warmup, then trace 5 batches
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(PROFILE_DIR),
        record_shapes=True,
        with_stack=True,
    )

    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | "
          f"{'Test Loss':>9} {'Test Acc':>8} | {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Only profile the first epoch
        use_profiler = profiler if epoch == 1 else None
        if use_profiler is not None:
            use_profiler.__enter__()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            profiler=use_profiler,
        )

        if use_profiler is not None:
            use_profiler.__exit__(None, None, None)
            print(f"  >> Profiler trace saved to {PROFILE_DIR}/")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.perf_counter() - t0
        print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:8.2f}% | "
              f"{test_loss:9.4f} {test_acc:7.2f}% | {elapsed:5.1f}s")

    print("\nTraining complete.")
    print(f"View profiler results:  tensorboard --logdir={PROFILE_DIR}")


if __name__ == "__main__":
    main()

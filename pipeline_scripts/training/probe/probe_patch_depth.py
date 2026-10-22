import argparse
from copy import deepcopy

import torch
from torch_em.loss import DiceLoss
from torch_em.model.unetr import get_unetr_model
from synapse_net.training.models import _get_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the peak GPU memory of one semisupervised training step for each patch depth z."
    )
    parser.add_argument("--backbone", type=str, default="sam2")
    parser.add_argument("--model_type", type=str, default="hvit_t_cells")
    parser.add_argument("--z", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 32])
    parser.add_argument("--xy", type=int, default=256)
    parser.add_argument("--separate_backward", action="store_true", default=False)
    return parser.parse_args()


def get_models(backbone, model_type):
    encoder_checkpoint = _get_checkpoint(backbone=backbone, model_type=model_type)
    model = get_unetr_model(
        ndim=3,
        backbone=backbone, model_type=model_type,
        encoder_checkpoint=encoder_checkpoint,
        out_channels=1, final_activation="Sigmoid",
    ).cuda()
    teacher = deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    return model, teacher


def forward_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def warmup_step(model, optimizer, scaler, loss_fn, shape):
    xs, ys = torch.rand(shape, device="cuda"), torch.rand(shape, device="cuda").round()

    optimizer.zero_grad()
    with forward_context():
        loss = loss_fn(model(xs), ys)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def mean_teacher_step(model, teacher, optimizer, scaler, loss_fn, shape, separate_backward):
    xs, ys = torch.rand(shape, device="cuda"), torch.rand(shape, device="cuda").round()
    xu1, xu2 = torch.rand(shape, device="cuda"), torch.rand(shape, device="cuda")

    optimizer.zero_grad()
    with forward_context():
        supervised_loss = loss_fn(model(xs), ys)
    if separate_backward:
        scaler.scale(supervised_loss / 2).backward()

    with forward_context(), torch.no_grad():
        pseudo_labels = teacher(xu1)

    with forward_context():
        unsupervised_loss = loss_fn(model(xu2), pseudo_labels)

    loss = unsupervised_loss / 2 if separate_backward else (supervised_loss + unsupervised_loss) / 2
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def measure_peak(step, *args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        step(*args)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    args = parse_args()
    model, teacher = get_models(args.backbone, args.model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.GradScaler("cuda")
    loss_fn = DiceLoss()

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{args.backbone} {args.model_type}, xy={args.xy}, separate_backward={args.separate_backward}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, {total:.1f} GB")
    print(f"{'z':>4} {'warmup':>10} {'mean_teacher':>14}")

    def fmt(peak):
        return "OOM" if peak is None else f"{peak:.2f} GB"

    for z in args.z:
        shape = (1, 1, z, args.xy, args.xy)
        warmup = measure_peak(warmup_step, model, optimizer, scaler, loss_fn, shape)
        mean_teacher = measure_peak(
            mean_teacher_step, model, teacher, optimizer, scaler, loss_fn, shape, args.separate_backward
        )
        print(f"{z:>4} {fmt(warmup):>10} {fmt(mean_teacher):>14}", flush=True)
        if mean_teacher is None:
            break


if __name__ == "__main__":
    main()

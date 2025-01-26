from lightning.pytorch.cli import LightningCLI
import torch


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False


def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

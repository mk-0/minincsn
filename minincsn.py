import math
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt


def forward(model, batch, noise_std):
    noise_std = noise_std[:, None, None, None]
    unit_noise = torch.randn_like(batch)
    noise = unit_noise * noise_std
    pertrubed = batch + noise
    prediction = model(pertrubed)
    mse = torch.pow(prediction - unit_noise, 2).sum(dim=[1, 2, 3])
    return mse.mean()


def langevin_dynamics(grad_fn, x, lr, steps, noise_scale=math.sqrt(2)):
    x = x.clone()
    for _step in range(steps):
        x += lr * grad_fn(x) + noise_scale * math.sqrt(lr) * torch.randn_like(x)
    return x


def sample(x, model, std_schedule, lr, steps_per_level, noise_scale=math.sqrt(2)):
    x = x.clone()
    path = [x.cpu().clamp(-1, 1)]
    with torch.no_grad():
        for std in std_schedule:
            grad_fn = lambda x: -std * model(x)
            noise_std = noise_scale * std
            x = langevin_dynamics(grad_fn, x, lr, steps_per_level, noise_std)
            path.append(x.cpu().clamp(-1, 1))
    return torch.stack(path, dim=1)


class MixerBlock(nn.Module):
    def __init__(self, patches, channels, inner_patches, inner_channels):
        super().__init__()
        self.tokenwise = nn.Sequential(
            nn.LayerNorm([channels, patches]),
            nn.Linear(patches, inner_patches),
            nn.GELU(),
            nn.Linear(inner_patches, patches),
        )
        self.channelwise = nn.Sequential(
            nn.LayerNorm([patches, channels]),
            nn.Linear(channels, inner_channels),
            nn.GELU(),
            nn.Linear(inner_channels, channels),
        )

    def forward(self, x):
        x = x + self.tokenwise(x)
        x = rearrange(x, "batch channel patch -> batch patch channel")
        x = x + self.channelwise(x)
        x = rearrange(x, "batch patch channel -> batch channel patch")
        return x


class MlpMixer(nn.Module):
    def __init__(
        self, patch_size, patches, channels, inner_patches, inner_channels, blocks
    ):
        super().__init__()
        self.first = nn.Conv2d(1, channels, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(patches, channels, inner_patches, inner_channels)
                for _ in range(blocks)
            ]
        )
        self.norm = nn.LayerNorm([channels, patches])
        self.last = nn.ConvTranspose2d(
            channels, 1, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.first(x)
        h = x.shape[2]
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.blocks(x)
        x = self.norm(x)
        x = rearrange(x, "b c (h w) -> b c h w", h=h)
        x = self.last(x)
        return x


class EMA:
    def __init__(self, state_dict, momentum=0.999):
        self.state_dict = {k: v.clone() for k, v in state_dict.items()}
        self.momentum = momentum

    def update(self, new_state_dict):
        with torch.no_grad():
            self.state_dict = {
                k: self.momentum * self.state_dict[k] + (1 - self.momentum) * v
                for k, v in new_state_dict.items()
            }


transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda img: img * 2 - 1),
    ]
)

reverse_transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: (img + 1) / 2),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.ToPILImage(),
    ]
)


if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(345345345)
    model = MlpMixer(4, 64, 128, 256, 512, 12).to(device)
    ema = EMA(model.state_dict())
    optim = torch.optim.Adam(model.parameters())
    std_schedule = torch.exp(torch.linspace(math.log(10), math.log(0.02), 150)).to(device)

    dataset = datasets.MNIST(
        root="MNIST/", train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(300):
        for batch, _label in (t := tqdm(loader, desc=f"Epoch {epoch:3}/300")):
            batch = batch.to(device)
            optim.zero_grad()
            level = torch.randint(0, len(std_schedule), [batch.shape[0]], device=device)
            loss = forward(model, batch, std_schedule[level])
            loss.backward()
            optim.step()
            ema.update(model.state_dict())
            t.postfix = f" loss={loss.item():.3f}"

    torch.save(ema.state_dict, f"ncsn_mnist.ckpt")
    ema_model = deepcopy(model)
    ema_model.load_state_dict(ema.state_dict)

    path = sample(
        torch.rand([8, 1, 32, 32], device=device) * 2 - 1,
        ema_model,
        std_schedule,
        lr=1 / 10,
        steps_per_level=10,
        noise_scale=math.sqrt(2),
    )
    grid = rearrange(path[:, ::10], "batch step ch h w -> ch (batch h) (step w)")
    plt.imshow(reverse_transform(-grid.clamp(-1, 1)), cmap="gray")
    plt.show()

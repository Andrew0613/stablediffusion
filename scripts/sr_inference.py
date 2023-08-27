import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
import argparse, os
import PIL
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from torch.utils import data as data
from torchvision import transforms

from scripts.txt2img import put_watermark
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.util import exists, instantiate_from_config

torch.set_grad_enabled(False)

class SingleImageDataset(data.Dataset):

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        self.lq_folder = opt.input_path

        self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):

        # load lq image
        lq_path = self.paths[index]
        img_lq = np.array(Image.open(lq_path))

        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

def init_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/x4-upscaling.yaml",
        nargs="?",
    )

    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained/x4-upscaler-ema.ckpt",
        nargs="?",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional, detailed, high-quality photo",
        help="the prompt to render"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="datasets/ClassicalSR/BSDS100/LRbicx4",
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps, minimum=2, maximum=200,",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples, min=1, max=4",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=0,
        help="eta of DDIM, min = 0, max = 1.0",
    )
    parser.add_argument(
        "--noise_level",
        type=int,
        default=50,
        help="Noise Augmentation, min=0, max = 350",
    )
    return parser.parse_args()


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device),
                         "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, init_size, num_samples=1, callback=None, eta=0., noise_level=None):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h, w)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(
                model, batch, noise_level)
            cond = {"c_concat": [x_augment],
                    "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [
                uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    result = result[:, :, 0:init_size[1] * 4, 0:init_size[0] * 4]
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded, pad_w, pad_h


def predict(input_image, sampler, prompt, steps, num_samples, scale, seed, eta, noise_level):
    init_image = input_image.convert("RGB")
    init_width, init_height = init_image.size
    image, pad_w, pad_h = pad_image(init_image)  # resize to integer multiple of 32
    width, height = image.size

    noise_level = torch.Tensor(
        num_samples * [noise_level]).to(sampler.model.device).long()
    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        seed=seed,
        scale=scale,
        h=height, w=width, steps=steps,
        init_size = [init_width, init_height],
        num_samples=num_samples,
        callback=None,
        noise_level=noise_level
    )

    return result

def main():

    opt = init_parser()
    test_dataset = SingleImageDataset(opt=opt)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    sampler = initialize_model(opt.config, opt.pretrained_model)
    to_pil_image = transforms.ToPILImage()

    os.makedirs(opt.save_path, exist_ok=True)

    for idx, input_data in enumerate(test_dataloader):
        input_image = to_pil_image(input_data['lq'].squeeze(dim=0).permute(2,0,1))
        results = predict(input_image, sampler, opt.prompt, opt.ddim_steps, opt.num_samples, opt.scale, opt.seed, opt.eta, opt.noise_level) 
        results[0].save(os.path.join(opt.save_path,input_data['lq_path'][0].split('/')[-1]))


if __name__ == "__main__":
    main()
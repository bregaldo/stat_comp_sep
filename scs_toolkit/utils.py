import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.utils import check_random_state


def imagenet_preprocess(input_image, no_mean=False):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406] if not no_mean else [0.0, 0.0, 0.0], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor

def imagenet_unnormalize(input):
    t = transforms.Compose([transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    return t(input)

def noise_psd(shape, batch_dim=False,  psd = lambda f: 1, ret_psd=False):
        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, )
        ndim = len(shape) - 1 if batch_dim else len(shape)
        data_shape = shape[1:] if batch_dim else shape
        for i in range(ndim):
            assert data_shape[i] == data_shape[0], "data_shape must be of the form (N, N, ...)"
        N = data_shape[0]

        # Build an array of isotropic wavenumbers
        wn = (np.fft.fftfreq(N)).reshape((N,) + (1,) * (ndim - 1))
        wn_iso = np.zeros(data_shape)
        for i in range(ndim):
            wn_iso += np.moveaxis(wn, 0, i) ** 2
        wn_iso = np.sqrt(wn_iso)
        S = psd(wn_iso)
        S = S / np.sqrt(np.mean(S**2)) # Normalize S

        X_white = np.fft.fftn(np.random.randn(*shape), axes=tuple(range(1, ndim + 1) if batch_dim else range(ndim)))
        X_shaped = X_white * S
        noises = np.fft.ifftn(X_shaped, axes=tuple(range(1, ndim + 1) if batch_dim else range(ndim))).real
        if ret_psd:
            return noises, S**2
        else:
            return noises

def PSDGenerator(f):
    return lambda shape, batch_dim=False, ret_psd=False: noise_psd(shape, batch_dim=batch_dim, psd=f, ret_psd=ret_psd)

@PSDGenerator
def white_noise(f):
    return np.ones_like(f)

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def pink_noise(f):
    return np.where(f == 0.0, 0.0, 1/np.sqrt(f))

def sparse_noise_batched(shape, fraction=.1, stencil=None, n_samples=10, random_state=None):
  rng = check_random_state(random_state)
  if stencil is None:
    stencil = np.array([[1.]])
  
  dirac_image = (rng.rand(n_samples, *shape) < fraction).astype('float32')

  total_padding = np.array(shape) - stencil.shape

  pad_after = total_padding // 2
  pad_before = total_padding - pad_after

  padded_stencil = np.pad(stencil, list(zip(pad_before, pad_after)))

  f_dirac_image = np.fft.fft2(dirac_image)
  f_padded_stencil = np.fft.fft2(np.fft.ifftshift(padded_stencil))

  f_noise = f_dirac_image * f_padded_stencil

  return np.fft.ifft2(f_noise).real

def get_noises(n, M, N, sigma, noise_batch_size_split=None, type='white', device=0, ret_psd=False):
    if type == 'white':
        ret = white_noise((n, M, N), batch_dim=True, ret_psd=ret_psd)
    elif type == 'pink':
        ret = pink_noise((n, M, N), batch_dim=True, ret_psd=ret_psd)
    elif type == 'blue':
        ret = blue_noise((n, M, N), batch_dim=True, ret_psd=ret_psd)
    elif type == 'crosses':
        stencil = np.zeros((6, 6))
        stencil[2:4] = 1
        stencil[:, 2:4] = 1
        ret = sparse_noise_batched((M, N), fraction=0.01, n_samples=n, stencil=stencil)
        if ret_psd:
            raise NotImplementedError
    else:
        raise ValueError("Unknown noise type")
    if ret_psd:
        noises, psd = ret
    else:
        noises = ret
    noises = torch.from_numpy(sigma*noises).to(device, dtype=torch.float32)
    out = ()
    if noise_batch_size_split is None:
        out += (noises,)
    else:
        out += (noises.view((noise_batch_size_split, n // noise_batch_size_split, M, N)),)
    if ret_psd:
        out += (sigma**2*psd,)
    if len(out) == 1:
        return out[0]
    else:
        return out

def get_exp_data(type, norm=True, torch=False, device=0, data_dir = '../data/', vgg=False):
    if type == "dust":
        s = np.load(os.path.join(data_dir, 'dust_data.npy')).astype(np.float32)
    elif type == "lss":
        s = np.load(os.path.join(data_dir, 'lss_data.npy')).astype(np.float32)
    elif type == "imagenet":
        input_image = Image.open(os.path.join(data_dir, 'imagenet_data.JPEG'))
        input_tensor = imagenet_preprocess(input_image).to(device)
        if vgg:
            s = input_tensor.unsqueeze(0).cpu().numpy()
        else:
            s = input_tensor.cpu().numpy().mean(axis=0)
    else:
        raise ValueError("Unknown data type")
    if norm:
        s = (s - s.mean()) / s.std()
    if torch:
        s = torch.from_numpy(s).to(device, dtype=torch.float32)
    return s

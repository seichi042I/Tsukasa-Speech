"""Model construction and checkpoint loading for Tsukasa Speech TTS."""

import torch
import torch.nn as nn
import yaml
from munch import Munch

from tsukasa_speech.utils.asr.models import ASRCNN
from tsukasa_speech.utils.jdc.model import JDCNet

from tsukasa_speech.diffusion.sampler import KDiffusion, LogNormalDistribution
from tsukasa_speech.diffusion.modules import Transformer1d, StyleTransformer1d
from tsukasa_speech.diffusion.diffusion import AudioDiffusionConditional

from tsukasa_speech.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from tsukasa_speech.models.architecture import (
    TextEncoder,
    ProsodyPredictor,
    StyleEncoder,
    LinearNorm,
)


def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()

    return F0_model


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'

    if args.decoder.type == "istftnet":
        from tsukasa_speech.vocoder.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size)
    else:
        from tsukasa_speech.vocoder.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes)

    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)

    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)

    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder

    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2,
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2,
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2,
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )

    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer


    nets = Munch(

            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor = pitch_extractor,

            mpd = MultiPeriodDiscriminator(),
            msd = MultiResSpecDiscriminator(),

            # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),

            # recon_diff = recon_diff,

       )

    return nets


def load_checkpoint(model, optimizer, path, load_only_params=False, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    print('loading the ckpt using the correct function.')

    for key in model:
        if key in params and key not in ignore_modules:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                print(f'{key} key length: {len(model[key].state_dict().keys())}, state_dict key length: {len(state_dict.keys())}')
                for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
            print('%s loaded' % key)

    if not load_only_params:
        iters = state.get("iters", 0)
        optimizer.load_state_dict(state["optimizer"])
    else:
        iters = 0

    return model, optimizer, 0, iters

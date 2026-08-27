import pytest
from functools import partial
from copy import deepcopy

import torch
from torch import nn, randint, randn, tensor, cuda

from einops import rearrange

import torch._dynamo
torch._dynamo.config.suppress_errors = True

cuda_available = cuda.is_available()

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    exists,
    stack_same_shape_tensors_with_inverse,
    filter_with_inverse,
    apply_fn_modality_type,
    SelfMaskedRepTraining
)
from transfusion_pytorch.modality_processing import ModalityItem

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('use_flex_attn', (False, True))
@pytest.mark.parametrize('reconstruction_loss_weight', (0., 0.1))
@pytest.mark.parametrize('model_output_clean', (False, True))
def test_transfusion(
    cache_kv: bool,
    use_flex_attn: bool,
    reconstruction_loss_weight: float,
    model_output_clean: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (16, 16),
        modality_default_shape = ((8,), (4,)),
        reconstruction_loss_weight = reconstruction_loss_weight,
        model_output_clean = model_output_clean,
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn
        ),
    )

    if use_flex_attn:
        model = model.cuda()

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    text_images_and_audio = [
        [randint_((16,)), (0, randn(4, 16)), randint_((8,)), (1, randn(6, 16))],
        [randint_((16,)), randn(7, 16), randint_((5,)), (1, randn(2, 16)), randint_((9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 4, cache_kv = cache_kv)


@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_auto_modality_transform(
    use_flex_attn: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (2, 2),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn
        )
    )

    text_and_images = [
        [randint_((16,)), randn(16, 2, 2), randint_((8,)), randn(16, 2, 2)],
        [randint_((16,)), randn(16, 2, 2), randint_((5,)), randn(16, 2, 2), randint_((9,))]
    ]

    loss = model(text_and_images)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 4)

@pytest.mark.parametrize('use_flex_attn', (False, True))
@pytest.mark.parametrize('return_loss', (False, True))
def test_text(
    use_flex_attn: bool,
    return_loss: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (8,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    text = randint(0, 256, (2, 64))

    model(text, return_loss = return_loss)

@pytest.mark.parametrize('channel_first', (False, True))
def test_modality_only(
    channel_first: bool
):

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (16, 16),
        channel_first_latent = channel_first,
        modality_default_shape = (8,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    images = randn(2, 4, 4, 16)

    if channel_first:
        images = rearrange(images, 'b ... d -> b d ...')

    loss = model(images, return_loss = True, modality_type = 1)

    loss.backward()

    model.generate_modality_only(modality_type = 1)

@pytest.mark.parametrize('custom_time_fn', (False, True))
def test_text_image_end_to_end(
    custom_time_fn: bool
):
    mock_vae_encoder = nn.Conv2d(3, 16, 3, padding = 1)
    mock_vae_decoder = nn.Conv2d(16, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 4,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = ((4, 4),),
        modality_encoder = mock_vae_encoder,
        modality_decoder = mock_vae_decoder,
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    text_and_images = [
        [
            randint(0, 4, (16,)),
            randn(3, 8, 8),
            randint(0, 4, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 4, (16,)),
            randn(3, 8, 5),
            randint(0, 4, (5,)),
            randn(3, 2, 16),
            randint(0, 4, (9,))
        ]
    ]

    # allow researchers to experiment with different time distributions across multiple modalities in a sample

    def num_modalities_to_times(num_modalities):
        batch = num_modalities.shape[0]
        device = num_modalities.device
        total_modalities = num_modalities.amax().item()
        return torch.ones((batch, total_modalities), device = device)

    time_fn = num_modalities_to_times if custom_time_fn else None

    # forward

    loss = model(
        text_and_images,
        num_modalities_to_times_fn = time_fn
    )

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(max_length = 4)

def test_velocity_consistency():
    mock_encoder = nn.Conv2d(3, 16, 3, padding = 1)
    mock_decoder = nn.Conv2d(16, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    ema_model = deepcopy(model)

    text_and_images = [
        [
            randint(0, 12, (16,)),
            randn(3, 8, 8),
            randint(0, 12, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 12, (16,)),
            randn(3, 8, 5),
            randint(0, 12, (5,)),
            randn(3, 2, 16),
            randint(0, 12, (9,))
        ]
    ]

    loss, breakdown = model(
        text_and_images,
        velocity_consistency_ema_model = ema_model,
        return_breakdown = True
    )

    loss.backward()

    assert exists(breakdown.velocity)

def test_axial_pos_emb():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (16, 16),                      # specify multiple latent dimensions
        modality_default_shape = ((2, 2), (2,)),    # default shapes for first and second modality
        fallback_to_default_shape_if_invalid = True,
        add_pos_emb = True,
        modality_num_dim = (2, 1),
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    # any torch.long is text, torch.float is modalities

    text_images_and_audio = [
        [randint(0, 256, (16,)), (0, randn(2, 3, 16)), randint(0, 256, (8,)), (1, randn(6, 16))],
        [randint(0, 256, (16,)), randn(1, 4, 16), randint(0, 256, (5,)), (1, randn(2, 16)), randint(0, 256, (9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(max_length = 4)

# unet related

def test_modality_only_with_unet():

    model = Transfusion(
        num_text_tokens = 10,
        dim_latent = 4,
        modality_default_shape = (14, 14),
        pre_post_transformer_enc_dec = (
            nn.Conv2d(4, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 3, 2, 1, output_padding = 1),
        ),
        channel_first_latent = True,
        add_pos_emb = True,
        modality_num_dim = 2,
        velocity_consistency_loss_weight = 0.1,
        transformer = dict(
            dim = 16,
            depth = 1,
            dim_head = 8,
            heads = 2
        )
    )

    x = torch.randn(1, 4, 14, 14)

    loss = model(x)
    loss.backward()

    sampled = model.generate_modality_only()

def test_stack_similar_shape_fn():
    from torch import zeros

    data = [
        zeros(3, 5),
        zeros(2, 3),
        zeros(3, 5),
        zeros(2, 3),
        zeros(4, 5),
        zeros(4, 5)
    ]

    plus_one = lambda x: x + 1

    data = [d + i for i, d in enumerate(data)]
    data_plus_one = [plus_one(d) for d in data]

    stacked_tensors, inverse = stack_same_shape_tensors_with_inverse(data)

    stacked_tensors = {k: plus_one(v) for k, v in stacked_tensors.items()}

    batch_processed_data_plus_one = inverse(stacked_tensors)

    assert all([torch.allclose(tensor1, tensor2) for tensor1, tensor2 in zip(data_plus_one, batch_processed_data_plus_one)])

def test_filter_with_inverse():
    x = [0, 1, 2, 3, 4]
    is_even = lambda el: (el % 2) == 0

    x_even, inverse = filter_with_inverse(is_even, x)
    x_even_times_ten = [el * 10 for el in x_even]

    y = inverse(x_even_times_ten)
    assert y == [0, 1, 20, 3, 40]

def test_apply_fn_modality_type():
    from torch import zeros

    modalities = [
        [zeros(3, 5)],
        [zeros(1, 5)],
        [(1, zeros(3, 5))],
        [(1, zeros(2, 5))],
        [(0, zeros(1, 5)), (1, zeros(3, 5))],
    ]

    modalities = apply_fn_modality_type(lambda x: x + 1, modalities)

    modalities = apply_fn_modality_type(lambda x: x + 2, modalities, modality_type = 1)

    assert (modalities[0][0][-1] == 1).all()
    assert (modalities[2][0][-1] == 2).all()


def test_zero_dimensional():

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 16,
        modality_default_shape = (),
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    # any torch.long is text, torch.float is modalities

    text_and_embeds = [
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (8,)), randn(16)],
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (5,)), randn(16), randint(0, 256, (9,))]
    ]

    loss = model(text_and_embeds)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(prompt = randn(16), max_length = 4)

def test_self_flow():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 16,
        modality_default_shape = (),
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    self_flow_wrapper = SelfMaskedRepTraining(
        model,
        use_asymmetric_dropout = True,
        student_dropout_rate = 0.1,
        teacher_dropout_rate = 0.,
        rep_loss_weight = 0.1,
        student_layer = -1,
        teacher_layer = -1,
    )

    text_and_embeds = [
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (8,)), randn(16)],
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (5,)), randn(16), randint(0, 256, (9,))]
    ]

    total_loss, (student_loss, self_flow_loss) = self_flow_wrapper(text_and_embeds)
    total_loss.backward()

    self_flow_wrapper.update_teacher()

    assert exists(self_flow_loss) and exists(student_loss)
    assert total_loss.shape == ()

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('prob_uncond', (0.0, 0.5, 1.0))
def test_classifier_free_guidance(
    cache_kv: bool,
    prob_uncond: float
):
    text_tokens = 16

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 8,
        prob_uncond = prob_uncond,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    # dataset batch with text and image modalities

    text_and_images = [
        [randint(0, text_tokens, (8,)), randn(4, 8), randint(0, text_tokens, (4,))],
        [randint(0, text_tokens, (6,)), randn(4, 8), randint(0, text_tokens, (5,))]
    ]

    # train forward pass with CFG drop

    model.train()
    loss = model(text_and_images)
    loss.backward()

    # sample with CFG scale > 1

    prompt = [randint(0, text_tokens, (4,))]
    sample_cfg1 = model.sample(prompt, max_length = 8, cfg_scale = 1.0, cache_kv = cache_kv)
    sample_cfg3 = model.sample(prompt, max_length = 8, cfg_scale = 3.0, cache_kv = cache_kv)

    assert len(sample_cfg1) > 0
    assert len(sample_cfg3) > 0

def test_e2e_multimodal_cfg_sampling():
    from transfusion_pytorch.transfusion import random_modality_length_to_time_fn

    text_tokens = 16

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 8,
        prob_uncond = 0.2,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    text_and_images = [
        [randint(0, text_tokens, (8,)), randn(4, 8), randint(0, text_tokens, (4,))],
        [randint(0, text_tokens, (6,)), randn(4, 8), randint(0, text_tokens, (5,))]
    ]

    loss = model(text_and_images, num_modalities_to_times_fn = random_modality_length_to_time_fn)
    loss.backward()

    # sample starting with a [som] token to force modality generation followed by text

    prime = [tensor([model.som_ids[0]])]
    sample = model.sample(prime, max_length = 16, cfg_scale = 2.5, cache_kv = True)

    assert len(sample) >= 3


def test_e2e_self_flow_with_cfg():
    model = Transfusion(
        num_text_tokens = 32,
        dim_latent = 16,
        prob_uncond = 0.1,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    wrapper = SelfMaskedRepTraining(
        model,
        use_asymmetric_dropout = True,
        student_dropout_rate = 0.1,
        teacher_dropout_rate = 0.,
        rep_loss_weight = 0.1,
        student_layer = -1,
        teacher_layer = -1
    )

    data = [
        [randint(0, 32, (12,)), randn(4, 16), randint(0, 32, (6,))]
    ]

    total_loss, (student_loss, self_flow_loss) = wrapper(data)
    total_loss.backward()
    wrapper.update_teacher()

    assert total_loss.ndim == 0

def test_generate_text_only():
    model = Transfusion(
        num_text_tokens = 256,
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

    prompt = torch.randint(0, 256, (1, 8))

    sampled_cached = model.generate_text_only(prompt, 24, temperature = 0., cache_kv = True)
    sampled_uncached = model.generate_text_only(prompt, 24, temperature = 0., cache_kv = False)

    assert sampled_cached.shape == (1, 16)
    assert torch.equal(sampled_cached, sampled_uncached)

def test_sample_cache_kv_equivalence():
    model = Transfusion(
        num_text_tokens = 256,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

    prompt = torch.randint(0, 256, (1, 8))

    torch.manual_seed(42)
    sampled_cached = model.sample(prompt = prompt, max_length = 16, cache_kv = True, text_temperature = 0.)

    torch.manual_seed(42)
    sampled_uncached = model.sample(prompt = prompt, max_length = 16, cache_kv = False, text_temperature = 0.)

    assert torch.equal(sampled_cached[0], sampled_uncached[0])

def test_e2e_multiple_modalities_interleaved():
    model = Transfusion(
        num_text_tokens = 16,
        dim_latent = (16, 16),
        modality_default_shape = ((4,), (3, 3)),
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

    mod0 = model.get_modality_info(0)
    mod1 = model.get_modality_info(1)

    prompt = [
        randint(0, 16, (3,)),
        tensor([model.meta_id]),
        model.char_tokenizer('4'),
        tensor([mod0.som_id]),
        (0, randn(4, 16)),
        tensor([mod0.eom_id]),
        randint(0, 16, (2,)),
        tensor([model.meta_id]),
        model.char_tokenizer('3,3'),
        tensor([mod1.som_id]),
        (1, randn(3, 3, 16)),
        tensor([mod1.eom_id]),
        randint(0, 16, (2,))
    ]

    torch.manual_seed(42)

    sampled_cached = model.sample(
        prompt = prompt,
        max_length = 10,
        cache_kv = True,
        text_temperature = 0.,
        modality_steps = 2
    )

    torch.manual_seed(42)

    sampled_uncached = model.sample(
        prompt = prompt,
        max_length = 10,
        cache_kv = False,
        text_temperature = 0.,
        modality_steps = 2
    )

    assert len(sampled_cached) == len(sampled_uncached)

    for item_cached, item_uncached in zip(sampled_cached, sampled_uncached):
        if isinstance(item_cached, tuple):
            type_c, tensor_c = (item_cached.modality_type, item_cached.tensor) if isinstance(item_cached, ModalityItem) else item_cached
            type_u, tensor_u = (item_uncached.modality_type, item_uncached.tensor) if isinstance(item_uncached, ModalityItem) else item_uncached

            assert type_c == type_u
            assert torch.allclose(tensor_c, tensor_u, atol = 1e-4)
        else:
            assert torch.equal(item_cached, item_uncached)

def test_processing_equivalence():
    # every modality processing strategy must produce identical results

    from transfusion_pytorch.modality_processing import assert_strategies_equivalent

    model = Transfusion(
        num_text_tokens = 8,
        dim_latent = 16,
        modality_default_shape = (4, 4),
        model_output_clean = False,
        transformer = dict(
            dim = 16,
            depth = 1
        )
    ).eval()

    batch = [
        [randint(0, 8, (7,)), (0, randn(4, 4, 16)), randint(0, 8, (11,)), randint(0, 8, (3,))],
        [randint(0, 8, (2,)), (0, randn(4, 4, 16)), randint(0, 8, (19,)), (0, randn(4, 4, 16)), randint(0, 8, (5,))]
    ]

    times = torch.ones(len(batch), 2)

    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    # decoding path (no meta tokens)

    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = False, return_embed = True)

def test_processing_channel_first_equivalence():
    # same, for channel first latents - exercises the per-instance singleton branch

    from transfusion_pytorch.modality_processing import assert_strategies_equivalent

    model = Transfusion(
        num_text_tokens = 8,
        dim_latent = 16,
        modality_default_shape = (4, 4),
        channel_first_latent = True,
        model_output_clean = False,
        transformer = dict(
            dim = 16,
            depth = 1
        )
    ).eval()

    batch = [
        [randint(0, 8, (7,)), (0, randn(16, 4, 4)), randint(0, 8, (11,)), (0, randn(16, 5, 6)), (0, randn(16, 4, 4))],
        [randint(0, 8, (2,)), (0, randn(16, 6, 6)), (0, randn(16, 4, 4)), randint(0, 8, (19,)), randint(0, 8, (5,))]
    ]

    times = torch.ones(len(batch), 3)

    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = True, return_embed = False)

# sample_many - batched sampling
# `sample_many` decodes a batch of samples concurrently, each walking the same state machine as
# `sample_one`, and must produce the same samples (modalities up to float noise, text exactly)

def make_sampling_model(
    num_modalities = 1,
    channel_first = False,
    **kwargs
):
    if num_modalities == 1:
        model = Transfusion(
            num_text_tokens = 16,
            dim_latent = 8,
            channel_first_latent = channel_first,
            modality_default_shape = (4, 4) if channel_first else (4,),
            transformer = dict(
                dim = 16,
                depth = 2,
                dim_head = 8,
                heads = 2
            ),
            **kwargs
        )
    else:
        model = Transfusion(
            num_text_tokens = 16,
            dim_latent = (8, 16),
            modality_default_shape = ((4,), (3, 3)),
            transformer = dict(
                dim = 16,
                depth = 2,
                dim_head = 8,
                heads = 2
            ),
            **kwargs
        )

    return model.eval()

def assert_sample_equivalence(model, prompt_batch, **kwargs):
    # `sample_many` must produce the same samples as `sample_one` run per prompt, with fixed
    # noise and greedy text decoding

    kwargs.setdefault('text_temperature', 0.)
    kwargs.setdefault('modality_steps', 4)

    cache_kv = kwargs.pop('cache_kv', True) # `sample_many` always uses the kv cache

    outs_one = [model.sample_one(prompt, cache_kv = cache_kv, **kwargs) for prompt in prompt_batch]
    outs_many = model.sample_many(prompt_batch, **kwargs)

    assert len(outs_one) == len(outs_many)

    for one, many in zip(outs_one, outs_many):
        assert len(one) == len(many)

        for one_part, many_part in zip(one, many):
            if isinstance(one_part, tuple):
                if isinstance(one_part, ModalityItem):
                    one_type, one_tensor = one_part.modality_type, one_part.tensor
                else:
                    one_type, one_tensor = one_part

                if isinstance(many_part, ModalityItem):
                    many_type, many_tensor = many_part.modality_type, many_part.tensor
                else:
                    many_type, many_tensor = many_part

                assert one_type == many_type
                assert torch.allclose(one_tensor, many_tensor, atol = 1e-4)
            else:
                assert torch.equal(one_part, many_part)

@pytest.mark.parametrize('cfg_scale', (1., 3.))
@pytest.mark.parametrize('channel_first', (False, True))
def test_sample_many_equivalent_to_sample_one(cfg_scale, channel_first):
    model = make_sampling_model(channel_first = channel_first)

    noise = torch.randn(32 if channel_first else 16, 8)
    prime = tensor([model.som_ids[0]])
    text_1 = randint(0, 16, (3,))
    text_2 = randint(0, 16, (2,))

    prompt_batches = [
        [[prime], [prime]],
        [[text_1], [text_2]],
        [[text_1], [text_2], [prime]]
    ]

    for prompt_batch in prompt_batches:
        assert_sample_equivalence(
            model,
            prompt_batch,
            init_modality_noise = noise,
            max_length = 16,
            cfg_scale = cfg_scale
        )

def test_sample_many_batched_multimodal():
    # heterogeneous batch - different modality types, shapes and latent dims, decoded
    # concurrently in one joint odeint trajectory

    model = make_sampling_model(num_modalities = 2)

    noise = torch.randn(32, 16)
    prime_0 = tensor([model.som_ids[0]])
    prime_1 = tensor([model.som_ids[1]])

    outs = model.sample_many(
        [[prime_0], [prime_1], [prime_0]],
        init_modality_noise = noise,
        max_length = 16,
        text_temperature = 0.,
        cfg_scale = 3.,
        modality_steps = 4
    )

    assert len(outs) == 3

    for sample in outs:
        assert isinstance(sample[1], tuple)
        assert sample[1][1].shape in ((4, 8), (3, 3, 16))

def test_sample_many_modality_prompt():
    model = make_sampling_model()

    img = randn(4, 8)

    outs = model.sample_many([(0, img), (0, img)], max_length = 10, text_temperature = 0., cfg_scale = 1.)

    assert len(outs) == 2

    for sample in outs:
        assert len(sample) == 3
        assert isinstance(sample[1], tuple)
        assert torch.allclose(sample[1][1], img)

def test_sample_many_encoder_decoder():
    mock_encoder = nn.Conv2d(3, 8, 3, padding = 1)
    mock_decoder = nn.Conv2d(8, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 16,
        dim_latent = 8,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

    img = randn(3, 8, 8)

    outs = model.sample_many([(0, img), (0, img)], max_length = 20, text_temperature = 0., cfg_scale = 1., modality_steps = 4)

    assert len(outs) == 2

    for sample in outs:
        assert isinstance(sample[1], tuple)
        assert sample[1][1].shape == (3, 8, 8)

def test_sample_many_empty_and_mixed_prompts():
    model = make_sampling_model()

    prime = tensor([model.som_ids[0]])

    outs = model.sample_many([None, None], max_length = 6, text_temperature = 0., cfg_scale = 1.)
    assert len(outs) == 2

    outs = model.sample_many([[None], [prime]], max_length = 12, text_temperature = 0., cfg_scale = 1., modality_steps = 4)
    assert len(outs) == 2
    assert isinstance(outs[1][1], tuple)

def test_sample_many_stochastic_text_distribution():
    # with temperature > 0 the sampled text should vary across runs

    model = make_sampling_model()

    prime = tensor([model.som_ids[0]])
    noise = torch.randn(16, 8)

    torch.manual_seed(0)
    outs_1 = model.sample_many([[prime]], init_modality_noise = noise, max_length = 20, text_temperature = 1.0, cfg_scale = 1., modality_steps = 4)

    torch.manual_seed(1)
    outs_2 = model.sample_many([[prime]], init_modality_noise = noise, max_length = 20, text_temperature = 1.0, cfg_scale = 1., modality_steps = 4)

    assert not torch.equal(outs_1[0][-1], outs_2[0][-1])

# force modality at start
# `force_modality_at_start` makes sampling skip text decoding and start off by decoding the given
# modality type, optionally with a forced shape - e.g. prompt with a video and force a 1d action
# of chunk length 32

def make_video_action_model():
    return Transfusion(
        num_text_tokens = 16,
        dim_latent = (16, 8),
        modality_default_shape = ((4, 4, 4), (16,)), # (video, action)
        channel_first_latent = (True, False),
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

def test_sample_force_modality_at_start():
    model = make_video_action_model()

    text = randint(0, 16, (3,))
    video = randn(16, 4, 4, 4) # channel first video latent - (latent dim, frames, height, width)

    sample = model.sample_one(
        [text, (0, video)], # text, then video
        force_modality_at_start = (1, (32,)), # force a 1d action of chunk length 32
        max_length = 64,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert isinstance(sample[1], tuple)
    assert sample[1][0] == 0
    assert torch.allclose(sample[1][1], video)

    assert isinstance(sample[3], tuple)
    assert sample[3][0] == 1
    assert sample[3][1].shape == (32, 8) # forced action of chunk length 32

    (sample_many_out,) = model.sample_many(
        [[text, (0, video)]],
        force_modality_at_start = (1, (32,)),
        max_length = 64,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert isinstance(sample_many_out[3], tuple)
    assert sample_many_out[3][0] == 1
    assert sample_many_out[3][1].shape == (32, 8)

def test_sample_force_modality_at_start_without_shape():
    # forcing only the modality type falls back to the modality's default shape

    model = make_sampling_model(num_modalities = 2)

    noise = torch.randn(16, 16)
    prime = tensor([model.som_ids[0]])

    outs = model.sample_many(
        [[prime]],
        force_modality_at_start = 1,
        init_modality_noise = noise,
        max_length = 32,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert len(outs) == 1

    sample = outs[0]

    assert isinstance(sample[1], tuple)
    assert sample[1][0] == 1
    assert sample[1][1].shape == (3, 3, 16) # default shape (3, 3) with latent dim 16

def test_sample_force_modality_at_start_heterogeneous_prompts():
    # a batch of prompts with different text and video lengths, all forced to decode the same
    # action shape

    model = make_video_action_model()

    text_1 = randint(0, 16, (2,))
    text_2 = randint(0, 16, (5,))

    video_1 = randn(16, 4, 4, 4) # 4 frames
    video_2 = randn(16, 2, 4, 4) # 2 frames
    video_3 = randn(16, 6, 4, 4) # 6 frames

    outs = model.sample_many(
        [
            [text_1, (0, video_1)],
            [text_2, (0, video_2)],
            [(0, video_3)],
        ],
        force_modality_at_start = (1, (32,)),
        max_length = 128,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert len(outs) == 3

    for sample in outs:
        assert isinstance(sample[-2], tuple)
        assert sample[-2][0] == 1
        assert sample[-2][1].shape == (32, 8) # same forced action shape

def test_sample_force_modality_at_start_equivalent_sample_one_many():
    # `sample_many` with a forced modality must produce the same samples as `sample_one` run per prompt

    model = make_video_action_model()

    text = randint(0, 16, (3,))
    video = randn(16, 4, 4, 4)
    noise = torch.randn(32, 8)

    assert_sample_equivalence(
        model,
        [[text, (0, video)], [text, (0, video)]],
        force_modality_at_start = (1, (32,)),
        init_modality_noise = noise,
        max_length = 64,
        cfg_scale = 1.
    )

# context only spans - a trailing bool flag on a text tensor or modality tuple marks it as
# context: attended to, but no loss on any of its tokens

def make_loss_masked_model(strategy = 'grouped', use_flex_attn = False):
    return Transfusion(
        num_text_tokens = 16,
        dim_latent = 16,
        modality_default_shape = (4, 4),
        modality_processing = strategy,
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2,
            use_flex_attn = use_flex_attn
        )
    ).eval()

def test_loss_masking():
    # the trailing weight of a sample item - `True` / `1` is the default, `False` / `0` gives no loss,
    # numbers are loss weights - demonstrated at the e2e level

    model = make_loss_masked_model()

    text = randint(0, 16, (8,))
    image = randn(4, 4, 16)

    def loss_of(sample, **kwargs):
        torch.manual_seed(0)
        return model(sample, times = tensor([[0.5]]), **kwargs)

    loss_default, breakdown_default = loss_of([[text, (0, image), text]], return_breakdown = True)

    # `True` and `1` are the same as the default

    assert torch.allclose(loss_of([[text, (0, image, True), text]]), loss_default)
    assert torch.allclose(loss_of([[text, (0, image, 1), text]]), loss_default)

    # a `0.5` weight halves the image's contribution to the loss

    loss_half = loss_of([[text, (0, image, 0.5), text]])

    assert 0. < loss_half < loss_default

    # `0` and `False` mask the image out of the loss - `False` also withholds it from attention

    _, breakdown_zero = loss_of([[text, (0, image, 0.), text]], return_breakdown = True)
    assert breakdown_zero.flow == []

    _, breakdown_false = loss_of([[text, (0, image, False), text]], return_breakdown = True)
    assert breakdown_false.flow == []
    assert not torch.equal(breakdown_false.text, breakdown_zero.text) # `False` excludes the span from attention - the suffix text conditions differently

    # a fully masked sample - no flow loss at all, only the [eos] target remains

    loss_masked, breakdown = loss_of([[(text, False), (0, image, False)]], return_breakdown = True)
    assert breakdown.flow == []
    assert loss_masked > 0.

def test_per_token_loss_masking():
    # a Float or Bool tensor on an item gives per-token loss weights over its tokens

    model = make_loss_masked_model()

    image = randn(4, 4, 16)

    def loss_of(sample, **kwargs):
        torch.manual_seed(0)
        return model(sample, times = tensor([[0.5]]), **kwargs)

    loss_default, _ = loss_of([[ (0, image) ]], return_breakdown = True)

    # an all-ones mask is the default

    assert torch.allclose(loss_of([[ (0, image, torch.ones(16)) ]]), loss_default)

    # the length is asserted

    with pytest.raises(AssertionError, match = 'per-token loss weights'):
        model([[(0, image, torch.ones(15))]])

    with pytest.raises(AssertionError, match = 'per-token loss weights'):
        model([[(randint(0, 16, (8,)), torch.ones(7))]])

    # Bool masks are accepted - only the masked tokens train

    half_mask = torch.linspace(0., 1., 16) > 0.5

    loss_bool, _ = loss_of([[(0, image, half_mask)]], return_breakdown = True)
    assert 0. < loss_bool < loss_default

    loss_zero, breakdown = loss_of([[(0, image, torch.zeros(16, dtype = torch.bool))]], return_breakdown = True)
    assert breakdown.flow == []
    assert loss_zero < loss_bool

def test_modal_loss_weight_specs():
    # every loss weight form on a modality in one batch - bool, float and per-token Bool / Float
    # tensors all work together

    model = make_loss_masked_model()

    images = [randn(4, 4, 16) for _ in range(6)]

    def loss_of(specs):
        batch = [[(0, image, spec)] for image, spec in zip(images, specs)]
        torch.manual_seed(0)
        return model(batch, times = tensor([[0.5]] * len(specs)), return_breakdown = True)

    half_mask = torch.linspace(0., 1., 16) > 0.5
    random_weights = torch.rand(16) * 0.5

    # a multidimensional list works too - flattened to the item's tokens

    grid_mask = [[1., 0., 0., 1.], [0., 1., 1., 0.], [1., 0., 1., 0.], [0., 1., 0., 1.]]

    mixed_specs = [True, 0.5, half_mask, random_weights, False, 0.]

    loss_mixed, breakdown_mixed = loss_of(mixed_specs)
    loss_defaults, _ = loss_of([1.0] * 6)

    assert breakdown_mixed.flow[0] > 0. # the weighted instances still train
    assert loss_mixed < loss_defaults   # the masked and weighted instances contribute less

    # a fully masked batch - no flow loss at all

    _, breakdown_masked = loss_of([False, 0., torch.zeros(16, dtype = torch.bool), torch.zeros(16), torch.zeros(16, dtype = torch.bool), torch.zeros(16)])
    assert breakdown_masked.flow == []

    # Bool and Float tensors of the same values are interchangeable, so are lists

    bool_specs = [half_mask, half_mask, torch.ones(16, dtype = torch.bool), grid_mask, torch.zeros(16, dtype = torch.bool), half_mask]
    float_specs = [half_mask.float(), half_mask.float(), torch.ones(16), tensor(grid_mask, dtype = torch.float), torch.zeros(16), half_mask.float()]

    _, breakdown_bool = loss_of(bool_specs)
    _, breakdown_float = loss_of(float_specs)

    assert torch.allclose(breakdown_bool.flow[0], breakdown_float.flow[0])

def test_context_only_spans_survive_encoder():
    # the weight on a modality must survive the encoders

    model = make_loss_masked_model()

    image = randn(4, 4, 16)

    out = apply_fn_modality_type(nn.Identity(), [(0, image, 0.5), (0, image)], modality_type = 0)

    assert len(out[0]) == 3 and out[0][2] == 0.5
    assert len(out[1]) == 2 and out[1][0] == 0

    assert model([[(0, image, 0.), (0, randn(4, 4, 16))]]).item() > 0.

def test_loss_masking_with_velocity_consistency():
    # velocity consistency with a masked demo in the batch - the ema pred flows stay aligned

    mock_encoder = nn.Conv2d(3, 16, 3, padding = 1)
    mock_decoder = nn.Conv2d(16, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    ema_model = deepcopy(model)

    text_and_images = [
        [
            randint(0, 12, (16,)),
            (0, randn(3, 8, 8), 0.), # masked demo - no loss, still attended
            randint(0, 12, (8,)),
            randn(3, 7, 7)
        ],
    ]

    loss, breakdown = model(
        text_and_images,
        velocity_consistency_ema_model = ema_model,
        return_breakdown = True
    )

    loss.backward()

    assert exists(breakdown.velocity)

def test_context_only_spans_flagged_prompts_sampling():
    # weights on prompt modalities are tolerated by sampling (everything in the prompt is context anyway)

    model = make_loss_masked_model()

    text = randint(0, 16, (3,))
    video = randn(4, 4, 16)

    sample = model.sample_one(
        [text, (0, video, True)],
        force_modality_at_start = (0, (4, 4)),
        max_length = 64,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert isinstance(sample[1], tuple)
    assert sample[1].modality_type == 0
    assert torch.allclose(sample[1].tensor, video)
    assert sample[1].loss_weight is True

    (sample_many_out,) = model.sample_many(
        [[text, (0, video, True)]],
        force_modality_at_start = (0, (4, 4)),
        max_length = 64,
        text_temperature = 0.,
        cfg_scale = 1.,
        modality_steps = 4
    )

    assert isinstance(sample_many_out[3], tuple)
    assert sample_many_out[3].modality_type == 0
    assert sample_many_out[3].tensor.shape == (4, 4, 16)

def test_loss_weight_edge_cases():
    # the remaining edges of the loss weight / context-only API: per-token weights on text, bare
    # `(tensor, weight)` specs, mixed weighted batches across processing and attention paths,
    # and weight specs during sampling

    model = make_loss_masked_model()
    text = randint(0, 16, (8,))
    image = randn(4, 4, 16)

    def loss_of(m, sample, **kwargs):
        torch.manual_seed(0)
        return m(sample, times = tensor([[0.5]]), **kwargs)

    # per-token weights on text - `(text_tensor, weights)`

    loss_default, _ = loss_of(model, [[text, (0, image)]], return_breakdown = True)
    loss_masked, _ = loss_of(model, [[(text, tensor([1., 0.] * 4)), (0, image)]], return_breakdown = True)

    assert 0. < loss_masked < loss_default

    with pytest.raises(AssertionError, match = 'per-token loss weights'):
        model([[(text, tensor([1.] * 9))]])

    # the bare `(tensor, weight)` form - no modality type - runs through a channel first encoder

    mock_encoder = nn.Conv2d(3, 8, 1)
    mock_decoder = nn.Conv2d(8, 3, 1)

    enc_model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 8,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(dim = 16, depth = 1)
    ).train()

    assert enc_model([[(randn(3, 4, 4), 0.5)]], times = tensor([[0.5]])).item() > 0.

    # weighted, per-token weighted and context-only items mix in one batch - every strategy agrees

    mixed_batch = [
        [(0, randn(4, 4, 16), 0.5), randint(0, 16, (5,)), (0, randn(4, 4, 16), False)],
        [(0, randn(4, 4, 16), tensor([0., 1.] * 8)), (randint(0, 16, (4,)), False)],
    ]

    mixed_times = tensor([[0.3, 0.6], [0.7, 0.8]])

    from transfusion_pytorch.modality_processing import assert_strategies_equivalent

    assert_strategies_equivalent(model, mixed_batch, mixed_times, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    if exists(flex_attention) and cuda_available:
        # both attention implementations agree on the loss for a context-only span

        flex_batch = [[(0, randn(4, 4, 16, device = 'cuda'), False), randint(0, 16, (5,), device = 'cuda')]]

        torch.manual_seed(0)
        loss_naive, _ = make_loss_masked_model().cuda()(flex_batch, times = tensor([[0.5]]), return_breakdown = True)

        torch.manual_seed(0)
        loss_flex, _ = make_loss_masked_model(use_flex_attn = True).cuda()(flex_batch, times = tensor([[0.5]]), return_breakdown = True)

        assert torch.allclose(loss_naive, loss_flex, atol = 1e-5)

    # weighted and zero weight prompt modalities are tolerated during sampling - naive and grouped

    for strategy in ('naive', 'grouped'):
        strategy_model = make_loss_masked_model(strategy)

        sample = strategy_model.sample_one(
            [text, (0, image, 0.5)],
            force_modality_at_start = (0, (4, 4)),
            max_length = 32,
            text_temperature = 0.,
            cfg_scale = 2.,
            modality_steps = 2
        )

        assert sample[1].modality_type == 0
        assert sample[1].loss_weight == 0.5

        zero_sample = strategy_model.sample_one(
            [(0, image, 0.)],
            force_modality_at_start = (0, (4, 4)),
            max_length = 32,
            text_temperature = 0.,
            cfg_scale = 1.,
            modality_steps = 2
        )

        assert zero_sample[1].loss_weight == 0.

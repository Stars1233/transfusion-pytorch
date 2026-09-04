from __future__ import annotations

"""
modality processing strategies for transfusion

  naive    - per-instance reference baseline
  grouped  - batches same (type, shape) groups
  flat     - batches whole modality type across varying shapes
  hybrid   - groups same shapes, falls back to flat for singletons
  auto     - (default) dynamically routes to fastest strategy for batch
"""

import math
import time
import statistics

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import torch
from torch import Tensor, tensor, is_tensor, cat, stack
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from einx import set_at

from torch_einops_utils import (
    pack_with_inverse,
    pad_at_dim,
    pad_left_at_dim,
    pad_sequence
)

# tensor typing (mirrors transfusion.py, kept local to avoid a circular import)

import jaxtyping

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)

# types

class ModalityItem(NamedTuple):
    # a modality in a sample, with named fields - used by the sampling API for returned samples

    modality_type: int
    tensor: Float['...']
    loss_weight: bool | float | Tensor | None = None

ItemLossWeight = float | Tensor

ModalitySample = list[
    Int[''] | Int['_'] | Float['...'] |
    tuple[int, Float['...']] | ModalityItem |
    tuple[Int[''], bool] | tuple[Float['...'], bool] |
    tuple[int, Float['...'], bool]
]

class ParsedItem(NamedTuple):
    # one raw batch entry normalized to its kind, tensor, modality type and trailing loss weight spec

    kind: str # 'text' or 'modality'
    tensor: Float['...']
    modality_type: int | None
    loss_weight: ItemLossWeight | None

GetPredFlows = dict[int, list[Callable[[Tensor], Tensor]]]

# small helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def is_int_tensor(t):
    return is_tensor(t) and t.dtype in (torch.int, torch.long)

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def add_temp_batch_dim(fn: Callable):
    def inner(t: Tensor, *args, **kwargs) -> Tensor:
        t = rearrange(t, '... -> 1 ...')
        out = fn(t, *args, **kwargs)
        out = rearrange(out, '1 ... -> ...')
        return out
    return inner

# decorator for model output to flow

def get_model_output_to_flow_fn(
    noised: Tensor,
    times: Tensor,
    eps = 5e-2,
    return_decorator = False
):
    if times.ndim == 0:
        times = rearrange(times, '-> 1')

    def to_flow(out):
        nonlocal noised
        noised = noised.reshape_as(out)
        padded_times = append_dims(times, out.ndim - 1)

        flow = (out - noised) / (1. - padded_times).clamp_min(eps)
        return flow

    if not return_decorator:
        return to_flow

    def decorator(fn):
        def inner(embed, *args, **kwargs):
            out = fn(embed, *args, **kwargs)
            return to_flow(out)
        return inner

    return decorator

@dataclass(eq = False)
class ModalityRecord:
    batch_index: int
    modality_type: int
    tensor: Tensor
    time: Tensor
    scatter_offset: int
    length: int
    axial_shape: tuple[int, ...]
    loss_weight: float | Tensor = 1.0
    not_attended: bool = False

class TextItem(NamedTuple):
    # a text chunk in a sample, normalized to its loss weight and whether it is attended to

    tensor: Int['n']
    loss_weight: ItemLossWeight = 1.0
    not_attended: bool = False

# one item in a sample, either a text chunk or a modality record - dispatched on `isinstance`

ScanItem = TextItem | ModalityRecord

class ProcessedModalityBatch(NamedTuple):
    text: Int['b n']
    modality_tokens: Float['b n d']
    modality_positions: list[list[tuple[int, int, int]]]
    modality_pos_emb: list | None
    flows: dict[int, list[Tensor]]
    get_pred_flows: GetPredFlows
    get_recon_losses: dict[int, list[Callable[[Tensor], Tensor]]]
    pos_emb_max_axial_dims: dict[int, list[Tensor]]
    total_tokens: int | None
    loss_weights: Tensor | None # per token loss weights over the text buffer - None when the batch is unmasked
    excluded: Tensor | None # padded (start, end) spans not attended to - None when nothing is excluded
    flow_weights: dict[int, list[float | Tensor]] # per instance flow loss weights, aligned with the list in `flows`

def parse_modality_item(item) -> ParsedItem:
    # normalize one batch item to its kind, tensor (warding a lone zero-dim text token to length 1),
    # modality type and trailing loss weight spec - a number or a per-token tensor / list when given

    def ward_scalar(tensor):
        return rearrange(tensor, '-> 1') if is_int_tensor(tensor) and tensor.ndim == 0 else tensor

    if isinstance(item, tuple):
        first, *rest = item

        if isinstance(first, int):
            # (modality_type, tensor[, loss_weight])

            tensor, weight = rest[0], rest[1] if len(rest) > 1 else None
            return ParsedItem('modality', ward_scalar(tensor), first, weight)

        # (tensor[, loss_weight])

        weight = rest[0] if rest else None
        kind = 'text' if is_int_tensor(first) else 'modality'
        return ParsedItem(kind, ward_scalar(first), 0 if kind == 'modality' else None, weight)

    kind = 'text' if is_int_tensor(item) else 'modality'
    return ParsedItem(kind, ward_scalar(item), 0 if kind == 'modality' else None, None)

def to_named_modality_item(item) -> ModalityItem | None:
    # normalize a modality sample entry into its named `ModalityItem`, or None when it is text -
    # `ModalityItem` inputs pass through the same parse, so everything is accessed uniformly

    parsed = parse_modality_item(item)

    if parsed.kind == 'text':
        return None

    return ModalityItem(parsed.modality_type, parsed.tensor, parsed.loss_weight)

def normalize_item_spec(spec) -> tuple[ItemLossWeight, bool]:
    # the trailing value of a sample item, normalized to (loss weight, not attended):
    #   `True` / `1`   - attend, weight 1 (the default when omitted)
    #   `False`        - not attended at all, no loss
    #   number         - loss weight over the whole item; `0` and `0.` give no loss, still attended
    #   Bool / Float tensor or list - per-token loss weights over the item's tokens, any shape
    #   (flattened), length asserted

    if not exists(spec):
        return 1.0, False

    if spec is True:
        return 1.0, False

    if spec is False:
        return 0.0, True

    if isinstance(spec, (int, float)):
        assert not isinstance(spec, bool)
        return float(spec), False

    if isinstance(spec, (list, tuple)):
        spec = tensor(spec)

    if is_tensor(spec):
        assert spec.dtype == torch.bool or spec.is_floating_point(), f'per-token loss weights must be Bool or Float, received {spec.dtype}'
        return spec.float().reshape(-1), False

    raise AssertionError(f'invalid loss weight for a sample item - must be a bool, number, or a Bool / Float tensor or list over the item tokens, received {spec}')

def weight_is_zero(weight: ItemLossWeight) -> bool:
    if is_tensor(weight):
        return bool(weight.sum() == 0.0)
    return weight == 0.0

def is_withheld(weight: ItemLossWeight, not_attended: bool) -> bool:
    # an item with a zero (or absent) loss weight - attended to or not - contributes no loss

    return not_attended or weight_is_zero(weight)

def validate_modality(modality_tensor: Tensor, modality_type: int, model) -> None:
    # check the modality sample against the modality info for that type

    assert 0 <= modality_type < model.num_modalities, f'received a modality index that is out of range. only {model.num_modalities} modalities specified'

    mod = model.get_modality_info(modality_type)
    channel_dim = 0 if mod.channel_first_latent else -1

    assert mod.dim_latent == modality_tensor.shape[channel_dim], f'mismatch for modality latent dimension - expected {mod.dim_latent} but received {modality_tensor.shape[-1]} - modality shape is {tuple(modality_tensor.shape)}, perhaps you need to set `channel_first_latent` to the correct value'
    assert mod.num_dim == (len(modality_tensor.shape) - 1), f'mismatch for modality number of dimensions - expected {mod.num_dim} but received {len(modality_tensor.shape) - 1} {modality_tensor.shape}'

def model_to_pred_flow(batch_index, start_index, modality_length, unpack_fn):
    # for parsing out the predicted flow from the flattened sequence of tokens coming out of the transformer

    def inner(embed: Float['b n d'], need_splice = True) -> Float['...']:
        embed = embed[batch_index]

        if need_splice:
            if embed.shape[0] < (start_index + modality_length):
                embed = embed[-modality_length:]
            else:
                embed = embed[start_index:(start_index + modality_length)]

        embed = unpack_fn(embed)
        return embed

    return inner

def weighted_token_loss(token_loss: Tensor, weights = None, channel_first = False) -> Tensor:
    weights = default(weights, 1.)

    if not is_tensor(weights):
        return token_loss.mean() * weights

    pattern = 'c ... -> c (...)' if channel_first else '... c -> (...) c'
    mask_pattern = 't -> 1 t' if channel_first else 't -> t 1'

    flat_loss = rearrange(token_loss, pattern)
    mask = rearrange(weights, mask_pattern)
    dim_latent = token_loss.shape[0 if channel_first else -1]

    return (flat_loss * mask).sum() / (mask.sum().clamp_min(1e-8) * dim_latent)

def get_recon_loss(noise, times, modality, weights = None, channel_first = False):
    def inner(pred_flow):
        recon = noise + pred_flow * (1. - times)
        return weighted_token_loss((recon - modality) ** 2, weights, channel_first)

    return inner

def get_recon_loss_lazy(noise, noised, times, shape, start, end, slice_, weights = None, channel_first = False):
    def inner(pred_flow):
        noise_instance = slice_(noise, start, end).reshape(shape)
        noised_instance = slice_(noised, start, end).reshape(shape)
        return get_recon_loss(noise_instance, times, noised_instance, weights, channel_first)(pred_flow)

    return inner

def group_records_by_shape(records) -> dict[tuple[int, ...], list[ModalityRecord]]:
    shape_groups = defaultdict(list)

    for record in records:
        shape_groups[record.axial_shape].append(record)

    return shape_groups

def scan_batch_for_structure(
    modalities: list[ModalitySample],
    times,
    model
) -> tuple[list[ModalityRecord], list[list[ScanItem]]]:
    # shared pass 1 - walk each sample for structure only, no gpu allocations in the hot path.
    # offsets, meta tokens and positions cannot be computed here: they depend on the modality
    # token lengths *after* the `latent_to_model` projection, which may downsample (unet style
    # encoders) - so they are computed in `assemble_batch` once the projection is done

    sample_items = []
    modality_records = []

    for batch_index, batch_modalities in enumerate(modalities):

        items: list[ScanItem] = []
        modality_index = 0

        for modality in batch_modalities:
            parsed = parse_modality_item(modality)
            loss_weight, not_attended = normalize_item_spec(parsed.loss_weight)

            # handle text

            if parsed.kind == 'text':
                chunk = parsed.tensor

                assert chunk.ndim == 1 and is_int_tensor(chunk)

                if is_tensor(loss_weight):
                    assert loss_weight.numel() == chunk.shape[0], f'per-token loss weights must match the number of text tokens ({chunk.shape[0]}), received {loss_weight.numel()}'

                items.append(TextItem(chunk, loss_weight, not_attended))
                continue

            # otherwise handle a modality
            # each modality instance gets its own noise level, indexed by its position in the sample

            modality_tensor = parsed.tensor
            modality_type = parsed.modality_type

            validate_modality(modality_tensor, modality_type, model)

            mod = model.get_modality_info(modality_type)

            modality_time = times[batch_index, modality_index]
            modality_index += 1

            axial_shape = modality_tensor.shape[1:] if mod.channel_first_latent else modality_tensor.shape[:-1]
            modality_length = math.prod(axial_shape)

            record = ModalityRecord(batch_index, modality_type, modality_tensor, modality_time, -1, modality_length, axial_shape, loss_weight, not_attended)

            modality_records.append(record)
            items.append(record)

        sample_items.append(items)

    return modality_records, sample_items

def get_cached_meta_tokens(model, device, shape_str, modality_type):
    # the constant [meta] [shape] [som] [eom] token tensors for a modality type + shape
    # string, cached per model + device (built once, not on every call)

    cache = getattr(model, '_modality_meta_cache', None)

    if not exists(cache):
        cache = model._modality_meta_cache = {}

    key = (str(device), shape_str, modality_type)

    if key not in cache:
        tensor_ = partial(tensor, device = device)
        mod = model.get_modality_info(modality_type)

        meta_tensor = tensor_([model.meta_id])
        shape_tokens = model.char_tokenizer(shape_str, device = device)
        som_tensor = tensor_([mod.som_id])
        eom_tensor = tensor_([mod.eom_id])

        cache[key] = (meta_tensor, shape_tokens, som_tensor, eom_tensor)

    return cache[key]

def assemble_batch(
    sample_items: list[list[ScanItem]],
    model,
    device,
    *,
    need_axial_pos_emb,
    return_embed
):
    # pass 2 - walk each sample again to compute the token offsets, meta tokens, positions
    # and axial positional embedding bookkeeping. the modality lengths and shapes used here
    # are the *projected* ones (updated on the records by the processing step), which for
    # downsampling (unet style) encoders differ from the raw input shapes

    text_chunks = [] # per sample, list of (offset, int tensor) to be scattered into the text buffer
    modality_positions = []
    modality_pos_emb = []
    weight_chunks = [] # per sample, list of (start, end, loss weight) spanning the text buffer
    excluded_spans = [] # per sample, list of (start, end) buffer spans not attended to
    pos_emb_max_axial_dims: dict[int, list[Tensor]] = defaultdict(list)

    total_lens = []

    for batch_index, items in enumerate(sample_items):

        offset = 0
        sample_text_chunks = []
        sample_modality_positions = []
        sample_modality_pos_emb = []
        sample_weight_chunks = []
        sample_excluded_spans = []

        for item in items:

            if isinstance(item, TextItem):
                chunk, weight, not_attended = item.tensor, item.loss_weight, item.not_attended

                sample_text_chunks.append((offset, chunk))

                if not_attended or is_tensor(weight) or weight != 1.0:
                    sample_weight_chunks.append((offset, offset + chunk.shape[0], 0. if not_attended else weight))

                offset += chunk.shape[0]

                if not_attended:
                    sample_excluded_spans.append((offset - chunk.shape[0], offset))

                if need_axial_pos_emb:
                    sample_modality_pos_emb.append(('zeros', chunk.shape[0]))

                continue

            # otherwise a modality instance

            record: ModalityRecord = item
            mod = model.get_modality_info(record.modality_type)

            precede_modality_tokens = succeed_modality_tokens = 0

            if not return_embed:
                # add the [meta] [shape] [som] ... [eom] tokens

                modality_shape_str = join([*map(str, record.axial_shape)], ',')
                meta_tensor, shape_tokens, som_tensor, eom_tensor = get_cached_meta_tokens(model, device, modality_shape_str, record.modality_type)

                precede_modality_tokens = len(shape_tokens) + 2
                succeed_modality_tokens = 1

                sample_text_chunks.extend((
                    (offset, meta_tensor),
                    (offset + 1, shape_tokens),
                    (offset + precede_modality_tokens - 1, som_tensor),
                    (offset + precede_modality_tokens + record.length, eom_tensor)
                ))

            scatter_offset = offset + precede_modality_tokens
            record.scatter_offset = scatter_offset

            sample_modality_positions.append((record.modality_type, scatter_offset, record.length))

            # per-token loss weights cover the modality tokens; a scalar weight covers the whole
            # span (meta tokens included), and `not_attended` items are excluded entirely

            weight = record.loss_weight

            if is_tensor(weight):
                assert weight.numel() == record.length, f'per-token loss weights must match the number of modality tokens ({record.length}), received {weight.numel()}'
                sample_weight_chunks.append((scatter_offset, scatter_offset + record.length, weight))
            elif weight != 1.0:
                sample_weight_chunks.append((offset, offset + record.length + precede_modality_tokens + succeed_modality_tokens, weight))

            if record.not_attended:
                sample_excluded_spans.append((offset, offset + record.length + precede_modality_tokens + succeed_modality_tokens))

            # handle axial positional embedding

            if need_axial_pos_emb:

                if exists(mod.pos_emb_mlp):
                    pos_emb_max_axial_dims[record.modality_type].append(tensor(record.axial_shape))
                    sample_modality_pos_emb.append((record.modality_type, record.axial_shape, (precede_modality_tokens, succeed_modality_tokens)))

                else:
                    sample_modality_pos_emb.append(('zeros', precede_modality_tokens + record.length + succeed_modality_tokens))

            offset += record.length + precede_modality_tokens + succeed_modality_tokens

        total_lens.append(offset)
        text_chunks.append(sample_text_chunks)
        modality_positions.append(sample_modality_positions)
        weight_chunks.append(sample_weight_chunks)
        excluded_spans.append(sample_excluded_spans)

        if need_axial_pos_emb:
            modality_pos_emb.append(sample_modality_pos_emb)

    return text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens, weight_chunks, excluded_spans

def build_loss_weights(weight_chunks, batch, max_len, device) -> Tensor | None:
    # per-sample chunks over the text buffer -> padded per-token loss weight grid, or None when all default

    if not any(weight_chunks):
        return None

    weights = torch.ones((batch, max_len), device = device)

    for batch_index, chunks in enumerate(weight_chunks):
        if not chunks:
            continue

        positions, values = [], []

        for start, end, weight in chunks:
            positions.append(torch.arange(start, end, device = device))

            if is_tensor(weight):
                values.append(weight.to(device = device, dtype = torch.float))
            else:
                values.append(torch.full((end - start,), weight, device = device))

        weights[batch_index] = set_at('[n], s [1], s -> [n]', weights[batch_index], rearrange(cat(positions), 's -> s 1'), cat(values))

    return weights

def build_excluded_spans(excluded_spans, device, dtype = torch.long) -> Tensor | None:
    # per-sample spans over the text buffer -> padded (start, end) coordinate tensor, or None when there are none

    if not any(excluded_spans):
        return None

    max_spans = max(len(spans) for spans in excluded_spans)

    padded = []

    for batch_spans in excluded_spans:
        spans = tensor(batch_spans, device = device, dtype = dtype) if batch_spans else torch.zeros((0, 2), device = device, dtype = dtype)
        # left pad rows of zero spans - a (0, 0) span never matches any position

        spans = pad_left_at_dim(spans, max_spans - spans.shape[0], dim = -2, value = 0)
        padded.append(spans)

    return stack(padded)

def process_modality_batch_naive(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # reference implementation, mirroring the original per-instance loop in `Transfusion.forward`

    device = model.device
    dim = model.dim

    tensor_ = partial(tensor, device = device)

    modality_positions = []
    modality_tokens = []
    modality_pos_emb = []
    weight_chunks = []
    excluded_spans = []

    text = []

    flows = defaultdict(list)

    get_pred_flows: GetPredFlows = defaultdict(list)

    get_recon_losses = defaultdict(list)

    flow_weights = defaultdict(list)

    pos_emb_max_axial_dims: dict[int, list[Tensor]] = defaultdict(list)

    for batch_index, batch_modalities in enumerate(modalities):

        modality_index = 0
        batch_modality_positions = []
        batch_modality_tokens = []
        batch_modality_pos_emb = []
        batch_weight_chunks = []
        batch_excluded_spans = []

        batch_text = []

        offset = 0

        for modality in batch_modalities:
            parsed = parse_modality_item(modality)
            weight, not_attended = normalize_item_spec(parsed.loss_weight)
            modality_tensor = parsed.tensor

            # handle text

            if parsed.kind == 'text':
                assert modality_tensor.ndim == 1 and is_int_tensor(modality_tensor)
                text_length = modality_tensor.shape[0]

                if is_tensor(weight):
                    assert weight.numel() == text_length, f'per-token loss weights must match the number of text tokens ({text_length}), received {weight.numel()}'

                batch_text.append(modality_tensor)
                zeros = torch.zeros(text_length, dim, device = device)

                batch_modality_tokens.append(zeros)

                if not_attended or is_tensor(weight) or weight != 1.0:
                    batch_weight_chunks.append((offset, offset + text_length, 0. if not_attended else weight))

                if not_attended:
                    batch_excluded_spans.append((offset, offset + text_length))

                offset += text_length

                if need_axial_pos_emb:
                    batch_modality_pos_emb.append(zeros)

                continue

            # otherwise handle a modality
            # each modality instance gets its own time column, indexed by its position in the sample

            modality_type = parsed.modality_type

            validate_modality(modality_tensor, modality_type, model)

            mod = model.get_modality_info(modality_type)

            modality_time = times[batch_index, modality_index]
            modality_index += 1

            participates_in_loss = not is_withheld(weight, not_attended)

            # noise

            if return_loss:
                noise = torch.randn_like(modality_tensor)

                noised_modality = modality_tensor * modality_time + noise * (1. - modality_time)

                # the flow is the (data - noise)

                modality_flow = modality_tensor - noise

                modality_tensor = noised_modality

                # items with a zero (or absent) loss weight - attended to, but no loss

                if participates_in_loss:
                    flows[modality_type].append(modality_flow)
                    flow_weights[modality_type].append(weight.to(device = device, dtype = torch.float) if is_tensor(weight) else weight)
                    get_recon_losses[modality_type].append(get_recon_loss(noise, modality_time, modality_tensor, weight, mod.channel_first_latent))

            # go through maybe encoder

            modality_tensor = add_temp_batch_dim(mod.latent_to_model)(modality_tensor)

            # gather the modality length

            modality_shape_tuple = modality_tensor.shape[:-1]
            modality_length = math.prod(modality_shape_tuple)

            if is_tensor(weight):
                assert weight.numel() == modality_length, f'per-token loss weights must match the number of modality tokens ({modality_length}), received {weight.numel()}'

            text_tensor = torch.full((modality_length,), -1, device = device) # text is all -1 here, so text labels are not learned on

            # only add modality meta information when not returning embedding, which only occurs when sampling modality

            succeed_modality_tokens = precede_modality_tokens = 0

            if not return_embed:
                # add the [som] and [eom] tokens for the modality type

                som_id, eom_id = mod.som_id, mod.eom_id

                # start by just storing the token length of the modality

                modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                modality_meta_info = model.char_tokenizer(modality_shape_str, device = device)

                precede_modality_tokens = len(modality_meta_info) + 2
                succeed_modality_tokens = 1

                text_tensor = cat((
                    tensor_([model.meta_id]),
                    modality_meta_info,
                    tensor_([som_id]),
                    text_tensor,
                    tensor_([eom_id])
                ))

            batch_modality_positions.append((modality_type, offset + precede_modality_tokens, modality_length)) # offset + preceding meta tag length (which includes the modality start token)

            # store parsing out back to shape

            modality_tensor, unpack_modality_shape = pack_with_inverse(modality_tensor, '* d')

            inverse_fn = model_to_pred_flow(batch_index, offset + precede_modality_tokens, modality_length, unpack_modality_shape)

            # maybe decorate the function if model output is predicting clean

            if model.model_output_clean:
                decorator = get_model_output_to_flow_fn(modality_tensor, modality_time, model.eps, return_decorator = True)
                inverse_fn = decorator(inverse_fn)

            # store function for extracting flow later

            if not return_loss or participates_in_loss:
                get_pred_flows[modality_type].append(inverse_fn)

            # the whole span (meta tokens included) is not attended and has no loss when excluded

            if not_attended:
                batch_excluded_spans.append((offset, offset + modality_length + precede_modality_tokens + succeed_modality_tokens))

            if is_tensor(weight):
                batch_weight_chunks.append((offset + precede_modality_tokens, offset + precede_modality_tokens + modality_length, weight))
            elif weight != 1.0:
                batch_weight_chunks.append((offset, offset + modality_length + precede_modality_tokens + succeed_modality_tokens, weight))

            # increment offset

            offset += modality_length + precede_modality_tokens + succeed_modality_tokens # +2 due to [som] and [eom] - then account for meta start id and modality shape information (or eventually any meta information about modality)

            modality_tensor = pad_at_dim(modality_tensor, (precede_modality_tokens, succeed_modality_tokens), dim = -2)

            batch_modality_tokens.append(modality_tensor)

            batch_text.append(text_tensor)

            # handle axial positional embedding

            if need_axial_pos_emb:

                if exists(mod.pos_emb_mlp):
                    pos_emb_max_axial_dims[modality_type].append(tensor(modality_shape_tuple))
                    pos_emb = (modality_type, modality_shape_tuple, (precede_modality_tokens, succeed_modality_tokens))

                else:
                    pos_emb = torch.zeros(text_tensor.shape[0], dim, device = device)

                batch_modality_pos_emb.append(pos_emb)

        text.append(cat(batch_text))

        if need_axial_pos_emb:
            modality_pos_emb.append(batch_modality_pos_emb)

        modality_tokens.append(cat(batch_modality_tokens))
        modality_positions.append(batch_modality_positions)
        weight_chunks.append(batch_weight_chunks)
        excluded_spans.append(batch_excluded_spans)

    total_tokens = sum([t.numel() for t in text]) if return_loss else None

    text = pad_sequence(text, value = -1)

    modality_tokens = pad_sequence(modality_tokens, dim = -2, value = 0.)

    loss_weights = build_loss_weights(weight_chunks, batch = len(modalities), max_len = text.shape[-1], device = device) if return_loss else None
    excluded = build_excluded_spans(excluded_spans, device) if return_loss else None

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text,
        modality_tokens = modality_tokens,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens,
        loss_weights = loss_weights,
        excluded = excluded,
        flow_weights = flow_weights
    )

class ProcessedRecord(NamedTuple):
    packed: Tensor
    noise: Tensor | None
    noised: Tensor | None
    flow: Tensor | None
    shape: tuple[int, ...] | None # original tensor shape, for reshaping flat noise slices back
    start: int | None
    end: int | None
    slice_: Callable | None # token-axis slicing function of the flat per-type tensors
    time: Tensor

def latent_projection_is_linear(module) -> bool:
    # whether a `latent_to_model` projection is elementwise (linear) over the token axis.
    # only such projections commute with concatenating instances along the token axis, which
    # the flat strategy relies on - conv / unet style encoders mix tokens across the
    # concatenation boundary, so flat must not be used for them

    if isinstance(module, (nn.Identity, nn.Linear, Rearrange)):
        return True

    if isinstance(module, nn.Sequential):
        return all(latent_projection_is_linear(m) for m in module)

    return False

def process_type_flat(records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process all instances of one modality type (of any shapes) as a single flat tensor:
    # one random noise, one noising, one latent projection for the whole type. the projected
    # tensor is always (S, d) so it slices along dim 0, but for channel first the noise /
    # noised / flow live in (c, S) so their token axis is dim 1. the flow targets keep their
    # flat slice shape - the flow loss packs them by value - and the recon loss closures
    # slice lazily, so there is no per-instance slicing work in the hot path (reconstruction
    # loss is off by default anyway)

    mod = model.get_modality_info(records[0].modality_type)
    channel_first = mod.channel_first_latent

    if not latent_projection_is_linear(mod.latent_to_model):
        # the projection mixes tokens across the concatenation boundary (conv / unet style
        # encoder), so the flat path is not applicable - process each instance directly

        return {record: process_instance(record, model, dim, return_loss) for record in records}

    # flatten each instance to its token sequence first - (length, d) for channel last,
    # (c, length) for channel first - then concatenate along the token axis

    def flatten(record):
        if channel_first:
            return record.tensor.reshape(record.tensor.shape[0], record.length)
        return record.tensor.reshape(record.length, record.tensor.shape[-1])

    combined = cat([flatten(record) for record in records], dim = 1 if channel_first else 0)

    if return_loss:
        times_ = stack([record.time for record in records])
        flat_times = times_.repeat_interleave(tensor([record.length for record in records], device = combined.device))

        if channel_first:
            flat_times = flat_times.view(1, -1)
        else:
            flat_times = append_dims(flat_times, 1)

        noise = torch.randn_like(combined)
        noised = combined * flat_times + noise * (1. - flat_times)
        flow = combined - noise
    else:
        noised = combined
        noise = flow = None

    # single latent projection for the whole type - `latent_to_model` is a linear over the
    # last dim, but expects a batch dim when channel first, so add one

    if channel_first:
        projected = mod.latent_to_model(noised[None, ...])[0]
    else:
        projected = mod.latent_to_model(noised)

    if channel_first:
        slice_ = lambda t, start, end: t[:, start:end]
    else:
        slice_ = lambda t, start, end: t[start:end]

    processed_by_record = {}

    offset = 0

    for record in records:
        start, end = offset, offset + record.length
        offset = end

        packed = projected[start:end].reshape(record.length, dim)

        if return_loss:
            processed_by_record[record] = ProcessedRecord(packed, noise, noised, slice_(flow, start, end), record.tensor.shape, start, end, slice_, record.time)
        else:
            processed_by_record[record] = ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)

    return processed_by_record

def process_group_stacked(shape_records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process a group of 2+ same-shaped instances with one batched noise, noising and projection

    stacked = stack([record.tensor for record in shape_records])

    if return_loss:
        times_ = stack([record.time for record in shape_records])
        padded_times = append_dims(times_, stacked.ndim - 1)

        noise = torch.randn_like(stacked)
        noised = stacked * padded_times + noise * (1. - padded_times)
        flow = stacked - noise
    else:
        noised = stacked
        noise = flow = None

    mod = model.get_modality_info(shape_records[0].modality_type)

    projected = mod.latent_to_model(noised) # single projection for the whole group

    processed_by_record = {}

    for ind, record in enumerate(shape_records):
        projected_instance = projected[ind]

        # the projection may downsample (unet style encoders) - the token length and axial
        # shape used for positions and the meta shape string are the *projected* ones

        record.length = math.prod(projected_instance.shape[:-1])
        record.axial_shape = tuple(projected_instance.shape[:-1])

        packed = projected_instance.reshape(record.length, dim)

        if return_loss:
            processed_by_record[record] = ProcessedRecord(packed, noise[ind], noised[ind], flow[ind], None, None, None, None, record.time)
        else:
            processed_by_record[record] = ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)

    return processed_by_record

def process_instance(record: ModalityRecord, model, dim, return_loss) -> ProcessedRecord:
    # process a single instance directly - a stack or cat would be a wasted copy

    mod = model.get_modality_info(record.modality_type)

    if return_loss:
        noise = torch.randn_like(record.tensor)
        noised = record.tensor * record.time + noise * (1. - record.time)
        flow = record.tensor - noise
    else:
        noised = record.tensor
        noise = flow = None

    # note: `latent_to_model` needs a batch dim when channel first

    if mod.channel_first_latent:
        projected = mod.latent_to_model(noised[None, ...])[0]
    else:
        projected = mod.latent_to_model(noised)

    # the projection may downsample (unet style encoders) - the token length and axial
    # shape used for positions and the meta shape string are the *projected* ones

    record.length = math.prod(projected.shape[:-1])
    record.axial_shape = tuple(projected.shape[:-1])

    packed = projected.reshape(record.length, dim)

    if return_loss:
        return ProcessedRecord(packed, noise, noised, flow, None, None, None, None, record.time)

    return ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)

def build_record_closures(
    records: list[ModalityRecord],
    processed_by_record: dict[ModalityRecord, ProcessedRecord],
    model,
    dim,
    modality_type,
    return_loss,
    flows,
    get_pred_flows,
    get_recon_losses,
    flow_weights
):
    # build the flow extraction functions and loss closures in scan order, so per type lists stay aligned

    for record in records:

        # items with a zero (or absent) loss weight - attended to or not - are excluded
        # from flow targets, pred flow closures and recon losses

        if return_loss and is_withheld(record.loss_weight, record.not_attended):
            continue

        processed = processed_by_record[record]

        # packing and unpacking a modality is a plain reshape - no per instance pack needed

        def unpack_fn(embed, shape = record.axial_shape, dim_ = dim):
            return embed.reshape(*shape, dim_)

        inverse_fn = model_to_pred_flow(record.batch_index, record.scatter_offset, record.length, unpack_fn)

        # maybe decorate the function if model output is predicting clean

        if model.model_output_clean:
            decorator = get_model_output_to_flow_fn(processed.packed, record.time, model.eps, return_decorator = True)
            inverse_fn = decorator(inverse_fn)

        get_pred_flows[modality_type].append(inverse_fn)

        if return_loss:
            flows[modality_type].append(processed.flow)

            weight = record.loss_weight.to(device = model.device, dtype = torch.float) if is_tensor(record.loss_weight) else record.loss_weight

            flow_weights[modality_type].append(weight)

            channel_first = model.get_modality_info(record.modality_type).channel_first_latent

            if exists(processed.slice_):
                # flat-style processing: the noise / noised live in the flat per-type tensor, slice lazily

                get_recon_losses[modality_type].append(get_recon_loss_lazy(processed.noise, processed.noised, processed.time, processed.shape, processed.start, processed.end, processed.slice_, weight, channel_first))

            else:
                get_recon_losses[modality_type].append(get_recon_loss(processed.noise, processed.time, processed.noised, weight, channel_first))

def process_type_grouped(records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process each (type, shape) group of 2+ instances with one batched noise, noising and
    # projection, and each singleton group per-instance

    processed_by_record = {}

    for shape_records in group_records_by_shape(records).values():

        if len(shape_records) > 1:
            # group of 2+ same-shaped instances - process all at once:
            # one noise, one noising, one projection for the whole group,
            # then scatter each result back into its position in the sequence

            processed_by_record.update(process_group_stacked(shape_records, model, dim, return_loss))

        else:
            # group of one - a stack would be a wasted copy, process the instance directly

            processed_by_record[shape_records[0]] = process_instance(shape_records[0], model, dim, return_loss)

    return processed_by_record

def process_type_hybrid(records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # best of both: same (type, shape) groups of 2+ are processed with one batched noise /
    # noising / projection (no concatenation copies), while all singleton groups of a type are
    # collected and processed together with the flat path (one noise / noising / projection
    # for the whole set) instead of per-instance.

    processed_by_record = {}

    shape_groups = group_records_by_shape(records)

    for shape_records in shape_groups.values():
        if len(shape_records) > 1:
            processed_by_record.update(process_group_stacked(shape_records, model, dim, return_loss))

    singleton_records = [shape_records[0] for shape_records in shape_groups.values() if len(shape_records) == 1]

    if singleton_records:
        processed_by_record.update(process_type_flat(singleton_records, model, dim, return_loss))

    return processed_by_record

def _process_modality_batch_with(
    process_type_fn: Callable,
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    device = model.device
    dim = model.dim

    batch = len(modalities)

    modality_records, sample_items = scan_batch_for_structure(
        modalities,
        times,
        model
    )

    # pass 2 - group all modality instances by modality type and process in parallel:
    # one random noise, one noising operation, one latent to model projection per group
    # (or per type for the flat path)

    records_by_type = defaultdict(list)

    for record in modality_records:
        records_by_type[record.modality_type].append(record)

    flows = defaultdict(list)
    get_pred_flows: GetPredFlows = defaultdict(list)
    get_recon_losses = defaultdict(list)
    flow_weights = defaultdict(list)

    processed_by_record = {}

    for modality_type, records in records_by_type.items():
        processed_by_record.update(process_type_fn(records, model, dim, return_loss))

    # pass 3 - compute the token offsets, meta tokens and positions from the *projected*
    # modality lengths, then build the flow extraction functions and loss closures in scan
    # order, so per type lists stay aligned

    text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens, weight_chunks, excluded_spans = assemble_batch(
        sample_items,
        model,
        device,
        need_axial_pos_emb = need_axial_pos_emb,
        return_embed = return_embed
    )

    for modality_type, records in records_by_type.items():
        build_record_closures(records, processed_by_record, model, dim, modality_type, return_loss, flows, get_pred_flows, get_recon_losses, flow_weights)

    # pass 4 - assemble each sample into a single pre-allocated buffer with per-chunk scatter,
    # replacing per-chunk allocations, padding and cats

    max_len = max(total_lens)

    loss_weights = build_loss_weights(weight_chunks, batch, max_len, device) if return_loss else None
    excluded = build_excluded_spans(excluded_spans, device) if return_loss else None

    text_bufs = torch.full((batch, max_len), -1, device = device)
    modality_bufs = torch.zeros((batch, max_len, dim), device = device)

    for batch_index, sample_chunks in enumerate(text_chunks):
        for offset, chunk in sample_chunks:
            text_bufs[batch_index, offset:(offset + chunk.shape[0])] = chunk

    for record in modality_records:
        packed = processed_by_record[record].packed
        modality_bufs[record.batch_index, record.scatter_offset:(record.scatter_offset + record.length)] = packed

    total_tokens = sum(total_lens) if return_loss else None

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text_bufs,
        modality_tokens = modality_bufs,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens,
        loss_weights = loss_weights,
        excluded = excluded,
        flow_weights = flow_weights
    )

def process_modality_batch(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:
    return _process_modality_batch_with(
        process_type_grouped,
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

def process_modality_batch_flat(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # per modality type, concatenate all instances (of any shape) into one tensor along the
    # token axis, then process the whole type with a single random noise, single noising and
    # single latent projection. the noise, noising and projection are all elementwise / linear
    # over the last dim, so nothing about the grouping by shape helps - this drops the kernel
    # count per type from one-per-group to a small constant. for non-linear projections (conv /
    # unet style encoders) `process_type_flat` falls back to per-instance processing.

    return _process_modality_batch_with(
        process_type_flat,
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

def process_modality_batch_hybrid(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:
    return _process_modality_batch_with(
        process_type_hybrid,
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

def evaluate_modality_pos_emb(
    modality_pos_emb,
    pos_emb_max_axial_dims,
    model,
    dim,
    device
):
    # lazily evaluate the modality positional embedding from the factorized positional embedding of maximum axial dims

    if not exists(modality_pos_emb):
        return None

    pos_emb_max_axial_dims = {mod_type: stack(sizes, dim = -1).amax(dim = -1) for mod_type, sizes in pos_emb_max_axial_dims.items()}
    factorized_pos_emb = {mod_type: model.get_modality_info(mod_type).pos_emb_mlp(max_size, return_factorized = True) for mod_type, max_size in pos_emb_max_axial_dims.items()}

    evaluated_pos_emb = []

    for batch_modality_pos_emb in modality_pos_emb:
        evaluated_batch_pos_emb = []

        for maybe_pos_emb_config in batch_modality_pos_emb:

            if is_tensor(maybe_pos_emb_config):
                evaluated_batch_pos_emb.append(maybe_pos_emb_config)
                continue

            if maybe_pos_emb_config[0] == 'zeros':
                _, length = maybe_pos_emb_config
                evaluated_batch_pos_emb.append(torch.zeros(length, dim, device = device))
                continue

            mod_type, mod_size, padding = maybe_pos_emb_config

            mod_info = model.get_modality_info(mod_type)
            mod_factorized_pos_emb = factorized_pos_emb[mod_type]

            mod_pos_emb = mod_info.pos_emb_mlp.combine_factorized(mod_factorized_pos_emb, mod_size, flatten = True)
            mod_pos_emb = pad_at_dim(mod_pos_emb, padding, dim = -2) # handle padding for preceding and succeeding meta tokens

            evaluated_batch_pos_emb.append(mod_pos_emb)

        evaluated_pos_emb.append(cat(evaluated_batch_pos_emb, dim = -2))

    return pad_sequence(evaluated_pos_emb, dim = -2, value = 0.)

# registry

PROCESSING_STRATEGIES = {
    'naive': process_modality_batch_naive,
    'grouped': process_modality_batch,
    'flat': process_modality_batch_flat,
    'hybrid': process_modality_batch_hybrid,
    'auto': None # set below once the auto router is defined
}

DEFAULT_PROCESSING_STRATEGY = 'auto'

# router - autodetect the fastest strategy for the batch structure at hand
# `'naive'` is excluded from routing: it is the reference baseline and never wins

ROUTING_CANDIDATES = ('grouped', 'flat', 'hybrid')

ROUTING_WARMUP = 1
ROUTING_ITERS = 3
ROUTING_MAX_CACHE = 64

def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def structure_signature(
    modalities: list[ModalitySample],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
):
    # cheap pure-python pass over the batch, mirroring `scan_batch_for_structure`'s element
    # interpretation - yields the cache key for the routing decision

    type_shape_counts = defaultdict(lambda: defaultdict(int))

    total_tokens = 0
    batch_size = 0

    for batch_modalities in modalities:
        batch_size += 1

        for modality in batch_modalities:
            parsed = parse_modality_item(modality)

            if parsed.kind == 'text':
                total_tokens += parsed.tensor.shape[0]
                continue

            mod = model.get_modality_info(parsed.modality_type)
            axial_shape = parsed.tensor.shape[1:] if mod.channel_first_latent else parsed.tensor.shape[:-1]

            total_tokens += math.prod(axial_shape)
            type_shape_counts[parsed.modality_type][axial_shape] += 1

    structure = tuple(
        (modality_type, shape, count)
        for modality_type in sorted(type_shape_counts)
        for shape, count in sorted(type_shape_counts[modality_type].items())
    )

    return (
        str(model.device),
        model.dim,
        batch_size,
        total_tokens,
        need_axial_pos_emb,
        return_loss,
        return_embed,
        structure
    )

class StrategyRouter:
    def __init__(
        self,
        candidates = ROUTING_CANDIDATES,
        warmup = ROUTING_WARMUP,
        iters = ROUTING_ITERS,
        max_cache = ROUTING_MAX_CACHE
    ):
        self.candidates = tuple(candidates)
        self.warmup = warmup
        self.iters = iters
        self.max_cache = max_cache
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def measure(self, modalities, times, model, *, need_axial_pos_emb, return_loss, return_embed):
        # time every candidate strategy on the actual batch and return the fastest

        device = model.device
        kwargs = dict(need_axial_pos_emb = need_axial_pos_emb, return_loss = return_loss, return_embed = return_embed)

        best = self.candidates[0]
        best_time = float('inf')

        for name in self.candidates:
            fn = get_processing_strategy(name)

            for _ in range(self.warmup):
                fn(modalities, times, model, **kwargs)

            samples = []

            for _ in range(self.iters):
                _sync_device(device)
                start = time.perf_counter()
                fn(modalities, times, model, **kwargs)
                _sync_device(device)
                samples.append(time.perf_counter() - start)

            median_time = statistics.median(samples)

            if median_time < best_time:
                best, best_time = name, median_time

        return best

    def route(self, modalities, times, model, *, need_axial_pos_emb, return_loss, return_embed):
        # pick the strategy for this batch - measured once per distinct batch structure, cached after

        key = structure_signature(
            modalities,
            model,
            need_axial_pos_emb = need_axial_pos_emb,
            return_loss = return_loss,
            return_embed = return_embed
        )

        if key in self.cache:
            return self.cache[key]

        if not key[-1]:
            # no modalities in the batch - every strategy is identical work, skip measuring
            strategy = 'hybrid'
        else:
            strategy = self.measure(
                modalities,
                times,
                model,
                need_axial_pos_emb = need_axial_pos_emb,
                return_loss = return_loss,
                return_embed = return_embed
            )

        self.cache[key] = strategy

        if len(self.cache) > self.max_cache:
            self.cache.pop(next(iter(self.cache))) # evict the oldest entry

        return strategy

ROUTER = StrategyRouter()

def process_modality_batch_auto(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # autodetect the fastest strategy for this batch structure (measured once, then cached)

    strategy = ROUTER.route(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

    return get_processing_strategy(strategy)(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

PROCESSING_STRATEGIES['auto'] = process_modality_batch_auto

def get_processing_strategy(name: str):
    assert name in PROCESSING_STRATEGIES, f'unknown modality processing strategy `{name}`, available: {list(PROCESSING_STRATEGIES)}'
    return PROCESSING_STRATEGIES[name]

def assert_strategies_equivalent(
    model,
    modalities,
    times,
    need_axial_pos_emb,
    return_loss,
    return_embed,
    strategy_names: list[str] | None = None
):
    # verify every strategy produces identical outputs (deterministic noise via mocked `torch.randn_like`)
    # used by the test suite and the benchmark before timing

    from unittest import mock

    strategy_names = default(strategy_names, list(PROCESSING_STRATEGIES))

    kwargs = dict(need_axial_pos_emb = need_axial_pos_emb, return_loss = return_loss, return_embed = return_embed)

    outputs = {}

    with mock.patch('torch.randn_like', side_effect = lambda t: torch.zeros_like(t)):
        for name in strategy_names:
            outputs[name] = get_processing_strategy(name)(modalities, times, model, **kwargs)

    reference = outputs[strategy_names[0]]

    for name in strategy_names[1:]:
        candidate = outputs[name]

        assert torch.equal(candidate.text, reference.text), f'{name}: text mismatch'
        assert torch.equal(candidate.modality_tokens, reference.modality_tokens), f'{name}: modality tokens mismatch'
        assert candidate.modality_positions == reference.modality_positions, f'{name}: positions mismatch'
        assert candidate.total_tokens == reference.total_tokens, f'{name}: total tokens mismatch'

        for mask_name in ('loss_weights', 'excluded'):
            mask_reference, mask_candidate = getattr(reference, mask_name), getattr(candidate, mask_name)
            assert (mask_candidate is None) == (mask_reference is None), f'{name}: {mask_name} presence mismatch'

            if exists(mask_reference):
                assert torch.equal(mask_candidate, mask_reference), f'{name}: {mask_name} mismatch'

        for modality_type in reference.flow_weights:
            weights_reference, weights_candidate = reference.flow_weights[modality_type], candidate.flow_weights[modality_type]

            assert len(weights_reference) == len(weights_candidate), f'{name}: flow weights length mismatch'

            for weight_reference, weight_candidate in zip(weights_reference, weights_candidate):
                weights_match = torch.equal(weight_reference, weight_candidate) if is_tensor(weight_reference) else weight_reference == weight_candidate
                assert weights_match, f'{name}: flow weights mismatch'

        embed = torch.randn(len(modalities), reference.text.shape[-1], model.dim, device = model.device)

        for modality_type in reference.get_pred_flows:
            assert modality_type in candidate.get_pred_flows, f'{name}: missing modality type {modality_type} in pred flows'

            for pred_reference, pred_candidate in zip(reference.get_pred_flows[modality_type], candidate.get_pred_flows[modality_type]):
                assert torch.allclose(pred_reference(embed), pred_candidate(embed)), f'{name}: pred flow closures mismatch'

        if return_loss:
            for modality_type in reference.flows:
                for flow_reference, flow_candidate in zip(reference.flows[modality_type], candidate.flows[modality_type]):
                    assert torch.equal(flow_reference.reshape(-1), flow_candidate.reshape(-1)), f'{name}: flow targets mismatch'

            for modality_type in reference.get_recon_losses:
                assert modality_type in candidate.get_recon_losses, f'{name}: missing modality type {modality_type} in recon losses'

                mod = model.get_modality_info(modality_type)

                for pred_fn, recon_ref, recon_cand in zip(reference.get_pred_flows[modality_type], reference.get_recon_losses[modality_type], candidate.get_recon_losses[modality_type]):
                    pred_flow = pred_fn(embed)
                    pred_flow = add_temp_batch_dim(mod.model_to_latent)(pred_flow)
                    assert torch.allclose(recon_ref(pred_flow), recon_cand(pred_flow), atol = 1e-5), f'{name}: recon loss closures mismatch'

    return outputs

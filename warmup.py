#!/usr/bin/env python3
"""Warmup script to load model and run test inference."""

import sys

sys.path.insert(0, '/app/CosyVoice')
sys.path.insert(0, '/app/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice3

print('Loading model for warmup...')
model = CosyVoice3(
    '/app/pretrained_models/Fun-CosyVoice3-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=True
)
print(f'Model loaded, sample rate: {model.sample_rate}')

# Quick inference to warm up all components
for output in model.inference_instruct2(
    tts_text='Test',
    instruct_text='You are a helpful assistant.<|endofprompt|>',
    prompt_wav=None,
    stream=False
):
    print(f'Warmup inference: {len(output["tts_speech"])} samples')
    break

print('Warmup complete')

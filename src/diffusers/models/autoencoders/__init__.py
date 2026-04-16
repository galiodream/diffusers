import importlib


def _optional_import(module_name, class_names):
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception:
        return

    for class_name in class_names:
        globals()[class_name] = getattr(module, class_name)


_optional_import("autoencoder_asym_kl", ["AsymmetricAutoencoderKL"])
_optional_import("autoencoder_dc", ["AutoencoderDC"])
_optional_import("autoencoder_kl", ["AutoencoderKL"])
_optional_import("autoencoder_kl_allegro", ["AutoencoderKLAllegro"])
_optional_import("autoencoder_kl_cogvideox", ["AutoencoderKLCogVideoX"])
_optional_import("autoencoder_kl_cosmos", ["AutoencoderKLCosmos"])
_optional_import("autoencoder_kl_flux2", ["AutoencoderKLFlux2"])
_optional_import("autoencoder_kl_hunyuan_video", ["AutoencoderKLHunyuanVideo"])
_optional_import("autoencoder_kl_hunyuanimage", ["AutoencoderKLHunyuanImage"])
_optional_import("autoencoder_kl_hunyuanimage_refiner", ["AutoencoderKLHunyuanImageRefiner"])
_optional_import("autoencoder_kl_hunyuanvideo15", ["AutoencoderKLHunyuanVideo15"])
_optional_import("autoencoder_kl_kvae", ["AutoencoderKLKVAE"])
_optional_import("autoencoder_kl_kvae_video", ["AutoencoderKLKVAEVideo"])
_optional_import("autoencoder_kl_ltx", ["AutoencoderKLLTXVideo"])
_optional_import("autoencoder_kl_ltx2", ["AutoencoderKLLTX2Video"])
_optional_import("autoencoder_kl_ltx2_audio", ["AutoencoderKLLTX2Audio"])
_optional_import("autoencoder_kl_magvit", ["AutoencoderKLMagvit"])
_optional_import("autoencoder_kl_mochi", ["AutoencoderKLMochi"])
_optional_import("autoencoder_kl_qwenimage", ["AutoencoderKLQwenImage"])
_optional_import("autoencoder_kl_temporal_decoder", ["AutoencoderKLTemporalDecoder"])
_optional_import("autoencoder_kl_wan", ["AutoencoderKLWan"])
_optional_import("autoencoder_longcat_audio_dit", ["LongCatAudioDiTVae"])
_optional_import("autoencoder_oobleck", ["AutoencoderOobleck"])
_optional_import("autoencoder_rae", ["AutoencoderRAE"])
_optional_import("autoencoder_tiny", ["AutoencoderTiny"])
_optional_import("autoencoder_vidtok", ["AutoencoderVidTok"])
_optional_import("consistency_decoder_vae", ["ConsistencyDecoderVAE"])
_optional_import("vq_model", ["VQModel"])

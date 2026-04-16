import importlib

from ...utils import is_torch_available


def _optional_import(module_name, class_names):
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception:
        return

    for class_name in class_names:
        globals()[class_name] = getattr(module, class_name)


if is_torch_available():
    _optional_import("auraflow_transformer_2d", ["AuraFlowTransformer2DModel"])
    _optional_import("cogvideox_transformer_3d", ["CogVideoXTransformer3DModel"])
    _optional_import("consisid_transformer_3d", ["ConsisIDTransformer3DModel"])
    _optional_import("dit_transformer_2d", ["DiTTransformer2DModel"])
    _optional_import("dual_transformer_2d", ["DualTransformer2DModel"])
    _optional_import("hunyuan_transformer_2d", ["HunyuanDiT2DModel"])
    _optional_import("latte_transformer_3d", ["LatteTransformer3DModel"])
    _optional_import("lumina_nextdit2d", ["LuminaNextDiT2DModel"])
    _optional_import("pixart_transformer_2d", ["PixArtTransformer2DModel"])
    _optional_import("prior_transformer", ["PriorTransformer"])
    _optional_import("sana_transformer", ["SanaTransformer2DModel"])
    _optional_import("stable_audio_transformer", ["StableAudioDiTModel"])
    _optional_import("t5_film_transformer", ["T5FilmDecoder"])
    _optional_import("transformer_2d", ["Transformer2DModel"])
    _optional_import("transformer_allegro", ["AllegroTransformer3DModel"])
    _optional_import("transformer_bria", ["BriaTransformer2DModel"])
    _optional_import("transformer_bria_fibo", ["BriaFiboTransformer2DModel"])
    _optional_import("transformer_chroma", ["ChromaTransformer2DModel"])
    _optional_import("transformer_chronoedit", ["ChronoEditTransformer3DModel"])
    _optional_import("transformer_cogview3plus", ["CogView3PlusTransformer2DModel"])
    _optional_import("transformer_cogview4", ["CogView4Transformer2DModel"])
    _optional_import("transformer_cosmos", ["CosmosTransformer3DModel"])
    _optional_import("transformer_easyanimate", ["EasyAnimateTransformer3DModel"])
    _optional_import("transformer_ernie_image", ["ErnieImageTransformer2DModel"])
    _optional_import("transformer_flux", ["FluxTransformer2DModel"])
    _optional_import("transformer_flux2", ["Flux2Transformer2DModel"])
    _optional_import("transformer_glm_image", ["GlmImageTransformer2DModel"])
    _optional_import("transformer_helios", ["HeliosTransformer3DModel"])
    _optional_import("transformer_hidream_image", ["HiDreamImageTransformer2DModel"])
    _optional_import("transformer_hunyuan_video", ["HunyuanVideoTransformer3DModel"])
    _optional_import("transformer_hunyuan_video15", ["HunyuanVideo15Transformer3DModel"])
    _optional_import("transformer_hunyuan_video_framepack", ["HunyuanVideoFramepackTransformer3DModel"])
    _optional_import("transformer_hunyuanimage", ["HunyuanImageTransformer2DModel"])
    _optional_import("transformer_kandinsky", ["Kandinsky5Transformer3DModel"])
    _optional_import("transformer_longcat_audio_dit", ["LongCatAudioDiTTransformer"])
    _optional_import("transformer_longcat_image", ["LongCatImageTransformer2DModel"])
    _optional_import("transformer_ltx", ["LTXVideoTransformer3DModel"])
    _optional_import("transformer_ltx2", ["LTX2VideoTransformer3DModel"])
    _optional_import("transformer_lumina2", ["Lumina2Transformer2DModel"])
    _optional_import("transformer_mochi", ["MochiTransformer3DModel"])
    _optional_import("transformer_nucleusmoe_image", ["NucleusMoEImageTransformer2DModel"])
    _optional_import("transformer_omnigen", ["OmniGenTransformer2DModel"])
    _optional_import("transformer_ovis_image", ["OvisImageTransformer2DModel"])
    _optional_import("transformer_prx", ["PRXTransformer2DModel"])
    _optional_import("transformer_qwenimage", ["QwenImageTransformer2DModel"])
    _optional_import("transformer_sana_video", ["SanaVideoTransformer3DModel"])
    _optional_import("transformer_sd3", ["SD3Transformer2DModel"])
    _optional_import("transformer_skyreels_v2", ["SkyReelsV2Transformer3DModel"])
    _optional_import("transformer_temporal", ["TransformerTemporalModel"])
    _optional_import("transformer_wan", ["WanTransformer3DModel"])
    _optional_import("transformer_wan_animate", ["WanAnimateTransformer3DModel"])
    _optional_import("transformer_wan_vace", ["WanVACETransformer3DModel"])
    _optional_import("transformer_z_image", ["ZImageTransformer2DModel"])

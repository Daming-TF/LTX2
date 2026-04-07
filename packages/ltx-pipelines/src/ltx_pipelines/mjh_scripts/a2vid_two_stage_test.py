"""
Run Example:
# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines/a2vid_two_stage.py \
    --checkpoint-path /root/autodl-tmp/huggingface/models/Lightricks--LTX-2/ltx-2-19b-dev.safetensors \
    --distilled-lora /root/autodl-tmp/huggingface/models/Lightricks--LTX-2/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /root/autodl-tmp/huggingface/models/Lightricks--LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /root/autodl-tmp/huggingface/models/google--gemma-3-12b-it-qat-q4_0-unquantized \
    --csv_file /root/autodl-tmp/mjh_proj/MoChaBench-main/benchmark/benchmark.xlsx \
    --source_dir /root/autodl-tmp/mjh_proj/MoChaBench-main/benchmark \
    --output_dir /root/autodl-tmp/outputs/ltx2/MochaBenchMark \
    --prompt "" \
    --output-path "" \
    --audio-path "" \
    --ref_image [Optional, include this flag if you want to use reference images for video generation]

# LTX 2.3
    --checkpoint-path /root/autodl-tmp/huggingface/models/Lightricks--LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /root/autodl-tmp/huggingface/models/Lightricks--LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /root/autodl-tmp/huggingface/models/Lightricks--LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --output_dir /root/autodl-tmp/outputs/ltx2_3/MochaBenchMark \

# View all available options for any pipeline
python -m ltx_pipelines.a2vid_two_stage --help
"""
import pathlib
from PIL import Image
import torch
import pandas as pd
import tqdm
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.a2vid_two_stage import (
    encode_video, 
    logging, 
    default_2_stage_arg_parser, 
    A2VidPipelineTwoStage, 
    TilingConfig, 
    get_video_chunks_number, 
    MultiModalGuiderParams,
    _probe_audio_duration_seconds,
    _align_num_frames,
)



@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to video duration (num_frames / frame_rate).",
    )
    # mjh's modify
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file containing test data for the pipeline.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ref_image",
        action="store_true",
    )
    ######
    args = parser.parse_args()
    pipeline = A2VidPipelineTwoStage(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    out_dir = pathlib.Path(args.output_dir)
    out_dir = out_dir/"wo_ref_image" if not args.ref_image else out_dir/"ref_image"
    if args.csv_file.lower().endswith((".xlsx", ".xls")):
        ds = pd.read_excel(args.csv_file)
    else:
        ds = pd.read_csv(args.csv_file)
    GREAN = '\033[92m'
    END = '\033[0m'
    process_bar = tqdm.tqdm(total=len(ds), desc="Processing rows")
    for _, row in ds.iterrows():
        idx_in_category = row['idx_in_category']
        category = row['category']
        context_id = row['context_id']
        args.prompt = row['prompt']

        ref_img_path = pathlib.Path(args.source_dir) / "first-frames-from-mocha-generation" / category / f"{context_id}.png"
        assert ref_img_path.exists(), f"Reference image not found at {ref_img_path}"
        w, h = Image.open(str(ref_img_path)).size
        args.images = [ImageConditioningInput(str(ref_img_path),0,0.8)]
        args.height = h//64*64
        args.width = w//64*64

        audio_path = pathlib.Path(args.source_dir) / "speeches" / category / f"{context_id}_speech.wav"
        # audio_path = pathlib.Path(args.source_dir) / "audios" / category / f"{context_id}.wav"
        assert audio_path.exists(), f"Audio file not found at {audio_path}"
        args.audio_path = str(audio_path)
        
        args.output_path = str(out_dir / category / f"{context_id}.mp4")
        pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"{GREAN}Processing{END} {category}/{context_id} \n {GREAN}Prompt:{END} {args.prompt}\n{GREAN}Output path:{END} {args.output_path}")


        ## auto-infer num_frames from audio duration when --num-frames is not explicitly provided, to avoid mismatch between video length and audio length which can cause quality degradation. Users can still specify --num-frames to override when needed.
        explicit_num_frames = "--num-frames" in sys.argv
        if not explicit_num_frames:
            target_duration = args.audio_max_duration
            if target_duration is None:
                target_duration = _probe_audio_duration_seconds(args.audio_path)

            if target_duration is not None and target_duration > 0:
                inferred_num_frames = _align_num_frames(target_duration, args.frame_rate)
                logging.info(
                    "\033[32m Auto num-frames enabled: \033[0m duration=%.3fs, fps=%.3f, inferred num_frames=%d",
                    target_duration,
                    args.frame_rate,
                    inferred_num_frames,
                )
                args.num_frames = inferred_num_frames
            else:
                logging.warning(
                    "\033[31m Auto num-frames failed to read audio duration from %s; using configured num_frames=%d \033[0m",
                    args.audio_path,
                    args.num_frames,
                )

        
        video, audio = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg_guidance_scale,
                stg_scale=args.video_stg_guidance_scale,
                rescale_scale=args.video_rescale_scale,
                modality_scale=args.a2v_guidance_scale,
                skip_step=args.video_skip_step,
                stg_blocks=args.video_stg_blocks,
            ),
            images=args.images if args.ref_image else [],
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
            audio_path=args.audio_path,
            audio_start_time=args.audio_start_time,
            audio_max_duration=args.audio_max_duration
            if args.audio_max_duration is not None
            else args.num_frames / args.frame_rate,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=args.max_batch_size,
        )

        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=video_chunks_number,
        )
        process_bar.update(1)


if __name__ == "__main__":
    main()
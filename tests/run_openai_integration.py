import os
import sys
import time
from pathlib import Path


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / '.env')

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print('Missing OPENAI_API_KEY in environment/.env', file=sys.stderr)
        sys.exit(2)

    import logging
    from openscenesense import VideoAnalyzer, ModelConfig, AnalysisPrompts, DynamicFrameSelector

    # Use latest OpenAI defaults; override explicitly for clarity
    models = ModelConfig(
        vision_model='gpt-4o',
        text_model='gpt-4o-mini',
        audio_model='whisper-1',  # segment-rich transcripts
    )

    prompts = AnalysisPrompts(
        frame_analysis=(
            "Describe the visible elements and actions in this frame."
        ),
        detailed_summary=(
            "Create a cohesive narrative that combines visual and audio elements from this {duration:.1f}-second video.\n"
            "Timeline:\n{timeline}\nTranscript:\n{transcript}"
        ),
        brief_summary=(
            "Provide a concise 2-3 line summary using timeline and transcript.\n"
            "Timeline:\n{timeline}\nTranscript:\n{transcript}"
        ),
    )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    analyzer = VideoAnalyzer(
        api_key=OPENAI_API_KEY,
        model_config=models,
        frame_selector=DynamicFrameSelector(),
        min_frames=6,
        max_frames=8,
        frames_per_minute=6.0,
        prompts=prompts,
        log_level=logging.INFO,
    )

    # Accept optional CLI arg for video path
    video_path = sys.argv[1] if len(sys.argv) > 1 else str(repo_root / 'Examples' / 'pizza.mp4')
    if not Path(video_path).exists():
        print(f'Missing test video: {video_path}', file=sys.stderr)
        sys.exit(3)

    t0 = time.time()
    results = analyzer.analyze_video(video_path)
    dt = time.time() - t0

    # Basic validations
    assert results.get('brief_summary'), 'Empty brief summary'
    assert results.get('summary'), 'Empty detailed summary'
    assert isinstance(results.get('frame_analyses'), list) and len(results['frame_analyses']) >= 1, 'No frame analyses'
    assert 'models_used' in results.get('metadata', {}), 'Missing models_used metadata'

    print('\n=== OpenAI Integration Test ===')
    print(f"Vision model: {models.vision_model}")
    print(f"Text model:   {models.text_model}")
    print(f"Audio model:  {models.audio_model}")
    print(f"Elapsed: {dt:.1f}s")

    print('\nBrief Summary:')
    print(results['brief_summary'])

    print('\nDetailed Summary (first 800 chars):')
    detailed = results['summary']
    print(detailed[:800] + ('...' if len(detailed) > 800 else ''))

    print('\nTimeline (first 20 lines):')
    timeline_lines = results['timeline'].splitlines()
    for line in timeline_lines[:20]:
        print(line)

    print('\nMetadata:')
    for k, v in results.get('metadata', {}).items():
        print(f'- {k}: {v}')

    # If audio segments exist, show a short transcript preview
    segs = results.get('audio_segments', [])
    if segs:
        print('\nTranscript preview (first 3 segments):')
        for seg in segs[:3]:
            print(f"[{seg['start_time']:.2f}-{seg['end_time']:.2f}] {seg['text']}")

    print('\nOK: OpenAI integration test completed successfully.')


if __name__ == '__main__':
    main()

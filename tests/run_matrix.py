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


def run_scenario(name, analyzer_ctor, kwargs, video_path):
    print(f"\n=== Scenario: {name} ===")
    t0 = time.time()
    try:
        analyzer = analyzer_ctor(**kwargs)
        results = analyzer.analyze_video(video_path)
        ok = (
            isinstance(results, dict)
            and 'timeline' in results
            and 'metadata' in results
            and 'frame_analyses' in results
        )
        print(f"OK: {ok}, frames={len(results.get('frame_analyses', []))}, segments={len(results.get('audio_segments', []))}")
        print(f"Summary present: {bool(results.get('summary'))}, Brief present: {bool(results.get('brief_summary'))}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    finally:
        print(f"Elapsed: {time.time() - t0:.1f}s")
    return True


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / '.env')

    pizza = str(repo_root / 'Examples' / 'pizza.mp4')
    gen   = str(repo_root / 'Examples' / 'genvideo.mp4')
    if not Path(pizza).exists() or not Path(gen).exists():
        print('Missing example videos; ensure Examples/pizza.mp4 and Examples/genvideo.mp4 exist.', file=sys.stderr)
        sys.exit(1)

    from openscenesense import (
        VideoAnalyzer,
        OpenRouterAnalyzer,
        ModelConfig,
        AnalysisPrompts,
        DynamicFrameSelector,
        UniformFrameSelector,
    )

    prompts = AnalysisPrompts()

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

    # Matrix runs (use small frame counts to reduce rate limiting)
    scenarios = []

    if OPENAI_API_KEY:
        # OpenAI defaults, no audio (pizza)
        scenarios.append((
            'OpenAI default + pizza (no audio) + Dynamic',
            VideoAnalyzer,
            dict(api_key=OPENAI_API_KEY,
                 model_config=ModelConfig(),
                 frame_selector=DynamicFrameSelector(),
                 min_frames=2, max_frames=3, frames_per_minute=2.0,
                 prompts=prompts,
                 max_workers=2),
            pizza
        ))

        # OpenAI defaults, with audio (gen)
        scenarios.append((
            'OpenAI default + gen (audio) + Dynamic',
            VideoAnalyzer,
            dict(api_key=OPENAI_API_KEY,
                 model_config=ModelConfig(),
                 frame_selector=DynamicFrameSelector(),
                 min_frames=2, max_frames=3, frames_per_minute=2.0,
                 prompts=prompts,
                 max_workers=2),
            gen
        ))

        # OpenAI non-Whisper audio to test JSON fallback
        scenarios.append((
            'OpenAI gpt-4o-mini-transcribe audio fallback + gen + Uniform',
            VideoAnalyzer,
            dict(api_key=OPENAI_API_KEY,
                 model_config=ModelConfig(audio_model='gpt-4o-mini-transcribe',
                                          vision_model='gpt-4o', text_model='gpt-4o-mini'),
                 frame_selector=UniformFrameSelector(),
                 min_frames=2, max_frames=3, frames_per_minute=2.0,
                 prompts=prompts,
                 max_workers=2),
            gen
        ))

    if OPENROUTER_API_KEY and OPENAI_API_KEY:
        # OpenRouter, minimal frames (pizza)
        scenarios.append((
            'OpenRouter free vision/text + pizza + Uniform',
            OpenRouterAnalyzer,
            dict(openrouter_key=OPENROUTER_API_KEY,
                 openai_key=OPENAI_API_KEY,
                 model_config=ModelConfig(
                     vision_model='qwen/qwen2.5-vl-32b-instruct:free',
                     text_model='meta-llama/llama-3.2-3b-instruct:free',
                     audio_model='whisper-1'),
                 frame_selector=UniformFrameSelector(),
                 min_frames=2, max_frames=3, frames_per_minute=2.0,
                 prompts=prompts,
                 max_workers=2),
            pizza
        ))

        # OpenRouter, with audio (gen)
        scenarios.append((
            'OpenRouter free vision/text + gen + Dynamic',
            OpenRouterAnalyzer,
            dict(openrouter_key=OPENROUTER_API_KEY,
                 openai_key=OPENAI_API_KEY,
                 model_config=ModelConfig(
                     vision_model='qwen/qwen2.5-vl-32b-instruct:free',
                     text_model='meta-llama/llama-3.2-3b-instruct:free',
                     audio_model='whisper-1'),
                 frame_selector=DynamicFrameSelector(),
                 min_frames=2, max_frames=3, frames_per_minute=2.0,
                 prompts=prompts,
                 max_workers=2),
            gen
        ))

    if not scenarios:
        print('No scenarios built. Ensure API keys are set in .env.', file=sys.stderr)
        sys.exit(2)

    passed = 0
    for name, ctor, kwargs, path in scenarios:
        if run_scenario(name, ctor, kwargs, path):
            passed += 1

    print(f"\nCompleted {len(scenarios)} scenarios, {passed} passed.")


if __name__ == '__main__':
    main()


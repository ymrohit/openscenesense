import logging
from openscenesense import ModelConfig,AnalysisPrompts,VideoAnalyzer,DynamicFrameSelector
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Example of using custom models
        custom_models = ModelConfig(
            vision_model="gpt-4o",  # Use any vision-capable model
            #vision_model="gpt-4o-mini",  # Use any vision-capable model
            text_model="gpt-4o-mini",  # Use any chat completion model
            audio_model="whisper-1"  # Use any Whisper model
        )

        # Custom prompts
        custom_prompts = AnalysisPrompts(
            frame_analysis="Analyze this frame focusing on visible elements, actions, and their relationship with any audio.",
            detailed_summary="""*DO NOT SEPARATE/MENTION AUDIO/VIDEO Specifically, MAKE IT COHESIVE AND NATURAL* Create a comprehensive narrative that cohesively integrates visual and audio elements into a single story or summary from this 
                       {duration:.1f}-second video:\n\nVideo Timeline:\n{timeline}\n\nAudio Transcript:\n{transcript}""",
            brief_summary="""Based on this {duration:.1f}-second video timeline and audio transcript:\n{timeline}\n\n{transcript}\n
                       Provide a concise cohesive short summary combining the key visual and audio elements, this should be easy to read and understand the entire context of the video *DO NOT SEPARATE/MENTION AUDIO/VIDEO Specifically, MAKE IT COHESIVE AND NATURAL* """

        )

        # Initialize analyzer
        analyzer = VideoAnalyzer(
            api_key="<OPENAI-API-KEY>",
            model_config=custom_models,
            min_frames=8,
            max_frames=32,
            frame_selector=DynamicFrameSelector(),
            frames_per_minute=8.0,
            prompts=custom_prompts,
            log_level=logging.INFO
        )

        # Analyze video
        video_path = "pizza.mp4"  # Replace with your video path
        results = analyzer.analyze_video(video_path)

        # Print results
        print("\nBrief Summary:")
        print("-" * 50)
        print(results['brief_summary'])

        print("\nDetailed Summary:")
        print("-" * 50)
        print(results['summary'])

        print("\nVideo Timeline:")
        print("-" * 50)
        print(results['timeline'])

        print("\nMetadata:")
        print("-" * 50)
        for key, value in results['metadata'].items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
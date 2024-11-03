# OpenSceneSense

**OpenSceneSense** is a cutting-edge Python package that revolutionizes video analysis by seamlessly integrating OpenAI and OpenRouter Vision models. Unlock the full potential of your videos with advanced frame analysis, audio transcription, dynamic frame selection, and comprehensive summaries all powered by state-of-the-art AI.

## Table of Contents

1. [🚀 Why OpenSceneSense?](#-why-openscenesense)
2. [🌟 Features](#-features)
3. [📦 Installation](#-installation)
4. [🔑 Setting Up API Keys](#-setting-up-api-keys)
5. [🛠️ Usage](#-usage)
6. [🎯 The Power of Prompts in OpenSceneSense](#-the-power-of-prompts-in-openscenesense)
7. [📈 Applications](#-applications)
8. [🚀 Future Upgrades: What's Next for OpenSceneSense?](#-future-upgrades-whats-next-for-openscenesense)
9. [🌐 OpenSceneSense and the Future of Content Moderation](#-openscenesense-and-the-future-of-content-moderation)
10. [🛠️ Contributing](#-contributing)
11. [📄 License](#-license)
12. [📬 Contact](#-contact)
13. [📄 Additional Resources](Docs/prompts.md)


## 🚀 Why OpenSceneSense?

OpenSceneSense isn't just another video analysis library, it's a gateway to a new era of video-based applications and innovations. By enabling large language models (LLMs) to process and understand video inputs, OpenSceneSense empowers developers, researchers, and creators to build intelligent video-centric solutions like never before.

### **Imagine the Possibilities:**

- **Interactive Video Applications:** Create applications that can understand and respond to video content in real-time, enhancing user engagement and interactivity.
- **Automated Video Content Generation:** Generate detailed narratives, summaries, or scripts based on video inputs, streamlining content creation workflows.
- **Advanced Video-Based Datasets:** Build rich, annotated video datasets for training and benchmarking machine learning models, accelerating AI research.
- **Enhanced Accessibility Tools:** Develop tools that provide detailed descriptions and summaries of video content, making media more accessible to all.
- **Smart Surveillance Systems:** Implement intelligent surveillance solutions that can analyze and interpret video feeds, detecting anomalies and providing actionable insights.
- **Educational Platforms:** Create interactive educational tools that can analyze instructional videos, generate quizzes, and provide detailed explanations.

With OpenSceneSense, the boundaries of what's possible with video analysis are limitless. Transform your ideas into reality and lead the charge in the next wave of AI-driven video applications.

## 🌟 Features

- **📸 Frame Analysis:** Utilize advanced vision models to dissect visual elements, actions, and their interplay with audio.
- **🎙️ Audio Transcription:** Seamlessly transcribe audio using Whisper models, enabling comprehensive multimedia understanding.
- **🔄 Dynamic Frame Selection:** Automatically select the most relevant frames to ensure meaningful and efficient analysis.
- **🔍 Scene Change Detection:** Identify scene transitions to enhance context awareness and narrative flow.
- **📝 Comprehensive Summaries:** Generate cohesive and detailed summaries that integrate both visual and audio elements.
- **🛠️ Customizable Prompts and Models:** Tailor the analysis process with custom prompts and model configurations to suit your specific needs.
- **📊 Metadata Extraction:** Extract valuable metadata for deeper insights and data-driven applications.

## 📦 Installation

### **Prerequisites**

- **Python 3.10+**
- **FFmpeg** installed on your system

### **Installing FFmpeg**

#### On Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS (using Homebrew)
```bash
brew install ffmpeg
```

#### On Windows
1. Download FFmpeg from [ffmpeg.org/download.html](https://ffmpeg.org/download.html).
2. Extract the archive.
3. Add the `bin` folder to your system PATH.

To verify the installation, run:
```bash
ffmpeg -version
```

### **Install OpenSceneSense**

```bash
pip install openscenesense
```

## 🔑 Setting Up API Keys

OpenSceneSense requires API keys for OpenAI and/or OpenRouter to access the AI models. You can set them as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Alternatively, you can pass them directly when initializing the analyzer in your code.

## 🛠️ Usage

### **Quick Start**

Get up and running with OpenSceneSense in just a few lines of code. Analyze your first video and unlock rich insights effortlessly.

```python
import logging
from openscenesense import ModelConfig, AnalysisPrompts, VideoAnalyzer, DynamicFrameSelector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up custom models and prompts
custom_models = ModelConfig(
    vision_model="gpt-4o",           # Vision-capable model
    text_model="gpt-4o-mini",        # Chat completion model
    audio_model="whisper-1"          # Whisper model for audio transcription
)

custom_prompts = AnalysisPrompts(
    frame_analysis="Analyze this frame focusing on visible elements, actions, and their relationship with any audio.",
    detailed_summary="""Create a cohesive narrative that integrates both visual and audio elements into a single summary. 
                        Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}""",
    brief_summary="""Provide a concise, easy-to-read summary combining the main visual and audio elements.
                     Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"""
)

# Initialize the video analyzer
analyzer = VideoAnalyzer(
    api_key="your-openai-api-key",
    model_config=custom_models,
    min_frames=8,
    max_frames=32,
    frame_selector=DynamicFrameSelector(),
    frames_per_minute=8.0,
    prompts=custom_prompts,
    log_level=logging.INFO
)

# Analyze the video
video_path = "path/to/your/video.mp4"
results = analyzer.analyze_video(video_path)

# Print the results
print("\nBrief Summary:")
print(results['brief_summary'])

print("\nDetailed Summary:")
print(results['summary'])

print("\nVideo Timeline:")
print(results['timeline'])

print("\nMetadata:")
for key, value in results['metadata'].items():
    print(f"{key}: {value}")
```

### **Advanced Usage with OpenRouter Models**

Leverage the power of OpenRouter models for enhanced performance and customization.

```python
from openscenesense import ModelConfig, AnalysisPrompts, OpenRouterAnalyzer, DynamicFrameSelector
from os import getenv

custom_models = ModelConfig(
    vision_model="meta-llama/llama-3.2-90b-vision-instruct:free",
    text_model="meta-llama/llama-3.2-3b-instruct:free",
    audio_model="whisper-1"
)

custom_prompts = AnalysisPrompts(
    frame_analysis="Analyze this frame focusing on visible elements, actions, and their relationship with any audio.",
    detailed_summary="""Create a cohesive narrative that integrates both visual and audio elements into a single summary. 
                        Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}""",
    brief_summary="""Provide a concise, easy-to-read summary combining the main visual and audio elements.
                     Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"""
)

analyzer = OpenRouterAnalyzer(
    openrouter_key=getenv("OPENROUTER_API_KEY"),
    openai_key=getenv("OPENAI_API_KEY"),
    model_config=custom_models,
    min_frames=8,
    max_frames=32,
    frame_selector=DynamicFrameSelector(),
    frames_per_minute=8.0,
    prompts=custom_prompts,
    log_level=logging.INFO
)

# Analyze the video
video_path = "path/to/your/video.mp4"
results = analyzer.analyze_video(video_path)

# Print the results
print("\nBrief Summary:")
print(results['brief_summary'])

print("\nDetailed Summary:")
print(results['summary'])

print("\nVideo Timeline:")
print(results['timeline'])

print("\nMetadata:")
for key, value in results['metadata'].items():
    print(f"{key}: {value}")
```

## 🎯 The Power of Prompts in OpenSceneSense

The quality and specificity of prompts play a crucial role in determining the effectiveness of the analysis OpenSceneSense provides. Thoughtfully crafted prompts can help guide the models to focus on the most important aspects of each frame, audio element, and overall video context, resulting in more accurate, relevant, and insightful outputs. OpenSceneSense allows you to define custom prompts for different types of analyses, giving you unparalleled control over the results.

### **Why Prompts Matter**

- **Directing Focus**: Prompts help guide the model’s attention to specific elements, such as actions, emotions, or interactions within the video.
- **Creating Coherent Summaries**: Well-defined prompts ensure that summaries are cohesive and natural, integrating both visual and audio information seamlessly.
- **Contextualizing with Metadata**: By including tags like `{timeline}`, `{duration}`, and `{transcript}`, prompts can encourage the model to generate outputs that are contextually aware, helping users understand the full scope of the video’s content.

### **Example Prompts for Enhanced Analysis**

Here are some example prompts to inspire you and help you maximize the capabilities of OpenSceneSense:

1. **Frame-Level Analysis Prompt**  
   ```plaintext
   "Analyze this frame with a focus on visible objects, their movements, and any emotions they convey. Consider the context of prior and subsequent frames for continuity."
   ```
   *Use this prompt to capture detailed visual elements and their implications.*

2. **Detailed Summary Prompt**  
   ```plaintext
   "*DO NOT SEPARATE AUDIO AND VIDEO SPECIFICALLY* Create a cohesive narrative that combines visual and audio elements naturally. Use context from the {duration:.1f}-second video with reference to the timeline:\n{timeline}\n\nAudio Transcript:\n{transcript}."
   ```
   *This prompt encourages the model to generate a unified story that reflects both the audio and visual content without separation.*

3. **Brief Summary Prompt**  
   ```plaintext
   "Provide a concise summary that combines key visual and audio elements. Base your answer on the {duration:.1f}-second video, using insights from the timeline:\n{timeline}\n\n{transcript}. This should be easy to read and provide the complete context."
   ```
   *Ideal for quickly understanding the main points of the video.*

4. **Emotion and Tone Analysis Prompt**  
   ```plaintext
   "Analyze the visual and audio tone of this video, noting any emotional shifts or significant interactions. Reference the timeline:\n{timeline} and audio transcript:\n{transcript} for a nuanced interpretation."
   ```
   *This prompt works well for assessing emotional tone or sentiment, especially in videos with spoken dialogue or expressive visuals.*

### **Using `{timeline}`, `{duration}`, and `{transcript}` Tags**

Including these tags in your prompts helps provide essential context to the models, resulting in richer, more accurate analyses:

- **`{timeline}`**: This tag allows the model to refer to specific points within the video, giving it the ability to track the progression of events and identify key moments.
- **`{duration}`**: By knowing the total duration, the model can gauge the significance of each scene and avoid overemphasizing minor moments.
- **`{transcript}`**: The audio transcript tag helps the model integrate spoken content into its interpretation, ensuring that the summary and analysis reflect both visual and audio insights.

### **Best Practices for Crafting Prompts**

- **Be Specific and Clear**: A focused prompt yields focused results. Specify whether the model should analyze actions, emotions, or the relationship between visuals and audio.
- **Use Descriptive Language**: The more descriptive your prompt, the better the model can interpret and analyze the content.
- **Integrate Tags for Full Context**: Use `{timeline}`, `{duration}`, and `{transcript}` in your prompts to enhance the model’s awareness of the video’s structure and narrative flow.

### **How Prompts and Tags Elevate OpenSceneSense**

By leveraging powerful prompts and contextual tags, OpenSceneSense can provide insights that feel human-like in their depth and coherence. These tailored prompts allow the model to interpret complex video content in a way that is both holistic and precise, setting OpenSceneSense apart as a tool for serious video analysis and understanding.

With prompt-driven analysis, OpenSceneSense can become your intelligent partner in interpreting video content, whether for content moderation, dataset creation, or building interactive applications that respond to visual and audio cues naturally.

For a comprehensive list of innovative video analysis prompts, refer to the [Prompt Examples](docs/InnovativeVideoAnalysisPrompts.md).

## 📈 Applications

OpenSceneSense is not just a tool—it's a foundation for building innovative video-centric solutions across various domains:

- **Media and Entertainment:** Automate content tagging, generate detailed video descriptions, and enhance searchability.
- **Education:** Develop intelligent tutoring systems that can analyze instructional videos and provide tailored feedback.
- **Healthcare:** Analyze medical procedure videos to assist in training and quality control.
- **Marketing:** Generate insightful video summaries and analytics to drive data-driven marketing strategies.
- **Research:** Create annotated video datasets for machine learning research, enabling advancements in computer vision and multimedia understanding.

## 🚀 Future Upgrades: What's Next for OpenSceneSense?

### **Top 5 Potential Future Upgrades**

1. **Real-Time Video Analysis**  
   Enabling real-time processing to analyze live video feeds with minimal latency. This would open doors to real-time content moderation, live video indexing, and intelligent surveillance systems that can act instantly based on video content.

2. **Multi-Language Audio and Text Support**  
   Expanding Whisper’s capabilities to transcribe and analyze videos in multiple languages, allowing OpenSceneSense to support a global user base and cater to diverse video content from around the world.

3. **Enhanced Metadata Extraction with Key Phrase Tagging**  
   Enabling automated tagging of key visual and audio elements as searchable metadata, which would improve video indexing and searchability, helping users find relevant content faster and more effectively.

---

## 🌐 OpenSceneSense and the Future of Content Moderation

With video content dominating the internet, content moderation is more crucial than ever. OpenSceneSense’s capabilities make it a groundbreaking tool for moderating content in an accurate, and context-aware way.

### **How OpenSceneSense Transforms Content Moderation**

- **Context-Aware Analysis:** Unlike traditional moderation methods that rely on keyword detection or basic image recognition, OpenSceneSense understands the full context by integrating both video and audio data. This enables it to distinguish between harmful and benign content with greater accuracy, reducing false positives.

- **Real-Time Moderation for Live Streams:** With future support for real-time analysis, OpenSceneSense could monitor live streams and flag inappropriate content immediately. This is essential for platforms hosting user-generated content where harmful material can spread quickly.

- **Automated Reporting and Summarization:** By generating detailed summaries and metadata, OpenSceneSense can quickly provide moderators with concise reports of flagged content, saving time and improving decision-making processes.

- **Cross-Cultural Sensitivity:** OpenSceneSense’s future multi-language and emotion recognition capabilities will allow it to identify culturally specific cues and context, making it a valuable tool for international platforms that need to moderate content with global sensibilities.

- **Safer Social Media and Video Platforms:** By empowering platforms with intelligent, context-aware moderation, OpenSceneSense will help create a safer online environment for users while reducing the burden on human moderators.

### **The Bottom Line**

As OpenSceneSense continues to evolve, its impact on content moderation will be transformative. It offers a way to analyze video content more holistically and sensitively than ever before, empowering platforms to ensure safer, more inclusive spaces for users worldwide.

## 🛠️ Contributing

I welcome contributions from the community! Whether it's reporting bugs, suggesting features, or submitting pull requests, your input helps make OpenSceneSense better for everyone.

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m "Add YourFeature"`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a pull request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

For questions, suggestions, or support, feel free to reach out:

- **Email:** mahendrarohittigon@gmail.com
- **GitHub Issues:** [OpenSceneSense Issues](https://github.com/ymrohit/openscenesense/issues)



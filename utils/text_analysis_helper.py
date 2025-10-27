"""
Helper functions for text analysis and metadata extraction
"""
import re
from typing import List
from outputModel.text_detection_output_model import TextMetadata, TextDetectionResult


def calculate_text_metadata(text: str) -> TextMetadata:
    """
    Calculate metadata for the analyzed text
    
    Args:
        text: Input text to analyze
        
    Returns:
        TextMetadata object with word count, character count, etc.
    """
    # Character count (including spaces)
    character_count = len(text)
    
    # Total size in characters (same as character_count)
    total_size = character_count
    
    # Word count (split by whitespace and filter empty strings)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (split by common sentence terminators)
    # This is a simple heuristic - might not be perfect for all cases
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings and whitespace-only strings
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    return TextMetadata(
        wordCount=word_count,
        totalSize=total_size,
        characterCount=character_count,
        sentenceCount=sentence_count
    )


def generate_text_detection_markdown(result: TextDetectionResult) -> str:
    """
    Generate markdown formatted report for text detection results
    Similar to how app_detect_image.py generates markdown
    
    Args:
        result: TextDetectionResult object with detection results
        
    Returns:
        Markdown formatted string
    """
    lines = []
    
    # Header
    lines.append("# AI Text Detection Results")
    lines.append("")
    
    # Detection Result
    lines.append(f"## Detection Result: `{result.prediction}`")
    lines.append("")
    lines.append(f"**Confidence**: `{result.confidence:.2%}`")
    lines.append(f"**Average Bot/AI Score**: `{result.average_bot_score:.4f}`")
    lines.append(f"**Average Human Score**: `{result.average_human_score:.4f}`")
    lines.append("")
    
    # Text Metadata
    if result.metadata:
        lines.append("### Text Metadata")
        lines.append("")
        lines.append(f"- **Word Count**: {result.metadata.wordCount}")
        lines.append(f"- **Character Count**: {result.metadata.characterCount}")
        lines.append(f"- **Total Size**: {result.metadata.totalSize} characters")
        lines.append(f"- **Sentence Count**: {result.metadata.sentenceCount}")
        lines.append("")
    
    # Detection Settings
    lines.append("### Detection Settings")
    lines.append("")
    lines.append(f"**Voting Method**: `{result.voting_method}`")
    lines.append("")
    
    # Voting breakdown
    lines.append("### Voting Results")
    lines.append("")
    if result.voting_breakdown:
        bot_votes = result.voting_breakdown.get('bot', 0)
        human_votes = result.voting_breakdown.get('human', 0)
        total_votes = bot_votes + human_votes
        lines.append(f"- **Bot/AI votes**: {bot_votes}")
        lines.append(f"- **Human votes**: {human_votes}")
        lines.append(f"- **Total models**: {total_votes}")
    lines.append("")
    
    # Individual Model Results
    if result.individual_scores:
        lines.append("### Individual Model Results")
        lines.append("")
        lines.append("| Model | Prediction | Confidence | Bot Score | Human Score |")
        lines.append("|-------|------------|------------|-----------|-------------|")
        
        for score in result.individual_scores:
            lines.append(
                f"| {score.model_name} | {score.prediction} | "
                f"{score.confidence:.4f} ({score.confidence*100:.1f}%) | "
                f"{score.bot_score:.4f} | {score.human_score:.4f} |"
            )
        
        lines.append("")
    
     
    lines.append("---")
    
    return "\n".join(lines)


def add_metadata_and_markdown(result: TextDetectionResult) -> TextDetectionResult:
    """
    Add metadata and markdown to a TextDetectionResult
    
    Args:
        result: TextDetectionResult object
        
    Returns:
        Updated TextDetectionResult with metadata and markdown populated
    """
    # Calculate metadata
    metadata = calculate_text_metadata(result.text)
    result.metadata = metadata
    
    # Generate markdown
    markdown = generate_text_detection_markdown(result)
    result.markdown = markdown
    
    return result

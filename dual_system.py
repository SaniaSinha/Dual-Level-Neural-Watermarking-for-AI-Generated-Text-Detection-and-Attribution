import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    DETECTOR_SAVE_PATH,
    MAX_LENGTH,
    AI_THRESHOLD,
    WATERMARK_BITS,
    DEFAULT_SOURCE_ID
)

from watermark import (
    SourceAwareWatermarkGenerator,
    StructurePhraseCodec,
    SemanticWatermarkCodec
)


class DualLevelWatermarkingSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_SAVE_PATH)
        self.detector_model = AutoModelForSequenceClassification.from_pretrained(
            DETECTOR_SAVE_PATH
        ).to(self.device)

        self.detector_model.eval()

        self.watermark_generator = SourceAwareWatermarkGenerator(
            total_bits=WATERMARK_BITS,
            source_bits=4
        )

        self.structure_codec = StructurePhraseCodec()
        self.semantic_codec = SemanticWatermarkCodec()

    def predict_ai_probability(self, text):
        inputs = self.detector_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.detector_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities[0][1].item()

    def detect_sentence(self, text):
        ai_probability = self.predict_ai_probability(text)

        return {
            "text": text,
            "ai_probability": ai_probability,
            "prediction": "AI-generated" if ai_probability >= AI_THRESHOLD else "Human-written"
        }

    def encode(self, text, text_id, source_id=DEFAULT_SOURCE_ID):
        ai_probability = self.predict_ai_probability(text)

        if ai_probability < AI_THRESHOLD:
            return {
                "success": False,
                "message": "Text classified as human-written. Watermarking skipped.",
                "ai_probability": ai_probability,
                "original_text": text
            }

        watermark_info = self.watermark_generator.generate_watermark(
            text_id=text_id,
            source_id=source_id
        )

        level1 = self.structure_codec.encode(
            text=text,
            watermark_bits=watermark_info["watermark"]
        )

        level2 = self.semantic_codec.encode(
            text=level1["watermarked_text"],
            remaining_bits=level1["remaining_bits"]
        )

        return {
            "success": True,
            "message": "Watermark embedded successfully.",
            "text_id": text_id,
            "ai_probability": ai_probability,
            "source_id": watermark_info["source_id"],
            "source_name": watermark_info["source_name"],
            "signature_bits": watermark_info["signature_bits"],
            "full_watermark": watermark_info["watermark"],
            "level1_bits": level1["level1_bits"],
            "level1_phrase": level1["selected_phrase"],
            "level2_bits": level1["remaining_bits"],
            "semantic_markers": level2["semantic_markers"],
            "watermarked_text": level2["final_watermarked_text"]
        }

    def decode(self, watermarked_text, text_id):
        level1 = self.structure_codec.decode(watermarked_text)
        level2 = self.semantic_codec.decode(watermarked_text)

        reconstructed_watermark = (
            level1["recovered_bits"] + level2["recovered_bits"]
        )[:WATERMARK_BITS]

        source_info = self.watermark_generator.decode_source(
            reconstructed_watermark
        )

        verification = self.watermark_generator.verify(
            text_id=text_id,
            recovered_watermark=reconstructed_watermark
        )

        return {
            "level1_detected": level1["detected"],
            "detected_phrase": level1["detected_phrase"],
            "level1_bits": level1["recovered_bits"],
            "semantic_markers": level2["detected_markers"],
            "level2_bits": level2["recovered_bits"],
            "reconstructed_watermark": reconstructed_watermark,
            "source_id": source_info["source_id"],
            "source_name": source_info["source_name"],
            "signature_bits": source_info["signature_bits"],
            "ownership_verified": verification["verified"],
            "verification_details": verification
        }

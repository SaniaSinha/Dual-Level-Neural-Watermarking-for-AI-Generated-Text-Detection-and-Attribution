import hashlib

from config import (
    SECURITY_KEY,
    WATERMARK_BITS,
    PHRASE_TABLE,
    SEMANTIC_TABLE,
    SOURCE_TABLE,
    DEFAULT_SOURCE_ID
)


class SourceAwareWatermarkGenerator:
    def __init__(self, security_key=SECURITY_KEY, source_table=SOURCE_TABLE, total_bits=16, source_bits=4):
        self.security_key = security_key
        self.source_table = source_table
        self.total_bits = total_bits
        self.source_bits = source_bits
        self.signature_bits = total_bits - source_bits

    def generate_signature_bits(self, text_id, source_id):
        data = f"{self.security_key}:{text_id}:{source_id}"
        hash_value = hashlib.sha256(data.encode()).hexdigest()
        binary_hash = bin(int(hash_value, 16))[2:].zfill(256)
        return binary_hash[:self.signature_bits]

    def generate_watermark(self, text_id, source_id=DEFAULT_SOURCE_ID):
        signature = self.generate_signature_bits(text_id, source_id)
        watermark = source_id + signature

        return {
            "source_id": source_id,
            "source_name": self.source_table[source_id],
            "signature_bits": signature,
            "watermark": watermark
        }

    def decode_source(self, recovered_watermark):
        source_id = recovered_watermark[:self.source_bits]
        signature_bits = recovered_watermark[self.source_bits:]

        return {
            "source_id": source_id,
            "source_name": self.source_table.get(source_id, "Unknown Source"),
            "signature_bits": signature_bits
        }

    def verify(self, text_id, recovered_watermark):
        decoded = self.decode_source(recovered_watermark)

        expected_signature = self.generate_signature_bits(
            text_id,
            decoded["source_id"]
        )

        return {
            "verified": expected_signature == decoded["signature_bits"],
            "source_id": decoded["source_id"],
            "source_name": decoded["source_name"],
            "expected_signature": expected_signature,
            "recovered_signature": decoded["signature_bits"]
        }


class StructurePhraseCodec:
    def __init__(self, phrase_table=PHRASE_TABLE):
        self.bits_to_phrase = phrase_table
        self.phrase_to_bits = {
            phrase: bits for bits, phrase in phrase_table.items()
        }

    def encode(self, text, watermark_bits):
        level1_bits = watermark_bits[:2]
        phrase = self.bits_to_phrase[level1_bits]

        return {
            "level1_bits": level1_bits,
            "selected_phrase": phrase,
            "watermarked_text": f"{phrase} {text}",
            "remaining_bits": watermark_bits[2:]
        }

    def decode(self, watermarked_text):
        for phrase, bits in self.phrase_to_bits.items():
            if watermarked_text.startswith(phrase):
                return {
                    "detected": True,
                    "detected_phrase": phrase,
                    "recovered_bits": bits,
                    "cleaned_text": watermarked_text[len(phrase):].strip()
                }

        return {
            "detected": False,
            "detected_phrase": None,
            "recovered_bits": "",
            "cleaned_text": watermarked_text
        }


class SemanticWatermarkCodec:
    def __init__(self, semantic_table=SEMANTIC_TABLE):
        self.bits_to_marker = semantic_table
        self.marker_to_bits = {
            marker: bits for bits, marker in semantic_table.items()
        }

    def split_bits_into_pairs(self, bits):
        return [
            bits[index:index + 2]
            for index in range(0, len(bits), 2)
            if len(bits[index:index + 2]) == 2
        ]

    def encode(self, text, remaining_bits):
        bit_pairs = self.split_bits_into_pairs(remaining_bits)
        semantic_markers = [
            self.bits_to_marker[pair]
            for pair in bit_pairs
        ]

        marker_sentence = (
            " Semantic watermark sequence: "
            + ", ".join(semantic_markers)
            + "."
        )

        return {
            "bit_pairs": bit_pairs,
            "semantic_markers": semantic_markers,
            "final_watermarked_text": text + marker_sentence,
            "embedded_bits": remaining_bits
        }

    def decode(self, watermarked_text):
        marker_header = "Semantic watermark sequence:"

        if marker_header not in watermarked_text:
            return {
                "detected_markers": [],
                "recovered_bits": ""
            }

        marker_section = watermarked_text.split(marker_header, 1)[1]

        normalized = (
            marker_section
            .replace(".", "")
            .replace(",", "")
            .replace(":", "")
            .lower()
        )

        recovered_bits = ""
        detected_markers = []

        for word in normalized.split():
            if word in self.marker_to_bits:
                recovered_bits += self.marker_to_bits[word]
                detected_markers.append(word)

        return {
            "detected_markers": detected_markers,
            "recovered_bits": recovered_bits
        }
